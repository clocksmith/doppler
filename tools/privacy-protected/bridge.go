package main

// bridge: the requester's key holder for a host process (Reploid) that runs the session protocol itself. The secret
// key is generated (or loaded from the key-epoch store) in this process and never leaves it. The bridge holds no
// weights and loads no model. It speaks over stdin/stdout to its parent only, never to an evaluator, and counts every
// encryption and decryption so the host can report them.
//
// Framing is protocol.go's: u32 header length | header JSON | blobs. Decrypted values return as one blob of
// little-endian float64s, so non-finite values survive (JSON cannot carry them) and the host decides what to do.

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

const bridgeProtocol = "doppler-protected-bridge/v1"

var bridgeCeiling = map[string]int64{"keygen": 0, "commit-keys": 0, "retire-epoch": 0, "encrypt": 0, "decrypt": maxResultBytes, "close": 0}

type stdioConn struct {
	r *bufio.Reader
	w *bufio.Writer
}

func (c *stdioConn) send(h header, blobs ...[]byte) error {
	h.Protocol = bridgeProtocol
	h.Blobs = nil
	for _, b := range blobs {
		h.Blobs = append(h.Blobs, int64(len(b)))
	}
	raw, err := json.Marshal(h)
	if err != nil {
		return err
	}
	var pre [4]byte
	binary.LittleEndian.PutUint32(pre[:], uint32(len(raw)))
	c.w.Write(pre[:])
	c.w.Write(raw)
	for _, b := range blobs {
		if _, err := c.w.Write(b); err != nil {
			return err
		}
	}
	return c.w.Flush()
}

func (c *stdioConn) recv() (header, [][]byte, error) {
	var h header
	var pre [4]byte
	if _, err := io.ReadFull(c.r, pre[:]); err != nil {
		return h, nil, err
	}
	n := binary.LittleEndian.Uint32(pre[:])
	if n == 0 || n > maxHeader {
		return h, nil, fmt.Errorf("header length %d outside (0, %d]", n, maxHeader)
	}
	raw := make([]byte, n)
	if _, err := io.ReadFull(c.r, raw); err != nil {
		return h, nil, err
	}
	if err := json.Unmarshal(raw, &h); err != nil {
		return h, nil, fmt.Errorf("malformed header: %w", err)
	}
	ceiling, ok := bridgeCeiling[h.Kind]
	if !ok || h.Protocol != bridgeProtocol {
		return h, nil, fmt.Errorf("bridge does not accept %q", h.Kind)
	}
	var total int64
	for _, b := range h.Blobs {
		if b < 0 {
			return h, nil, fmt.Errorf("negative blob length")
		}
		total += b
	}
	if total > ceiling {
		return h, nil, fmt.Errorf("%s carries %d bytes, above its %d ceiling", h.Kind, total, ceiling)
	}
	blobs := make([][]byte, len(h.Blobs))
	for i, b := range h.Blobs {
		blobs[i] = make([]byte, b)
		if _, err := io.ReadFull(c.r, blobs[i]); err != nil {
			return h, nil, err
		}
	}
	return h, blobs, nil
}

func bridgeMain(args []string) error {
	fs := flag.NewFlagSet("bridge", flag.ExitOnError)
	profileID := fs.String("profile", "research-n16", "pinned profile")
	allowInsecure := fs.Bool("allow-insecure-dev", false, "permit the insecure development profile")
	stateDir := fs.String("state-dir", "", "requester key-epoch store (secret key, 0600); reused until it expires")
	keyTTL := fs.Duration("key-ttl", 24*time.Hour, "lifetime of a new key epoch")
	fs.Parse(args)
	prof, err := lookupProfile(*profileID)
	if err != nil {
		return err
	}
	if prof.Insecure && !*allowInsecure {
		return fmt.Errorf("profile %s is insecure and only allowed with --allow-insecure-dev", prof.ID)
	}
	params, err := prof.params()
	if err != nil {
		return err
	}
	o := requesterOptions{prof: prof, stateDir: *stateDir, keyTTL: *keyTTL}
	ep, sk, reused, err := loadOrCreateEpoch(o, params)
	if err != nil {
		return err
	}
	r := &requester{params: params, sk: sk, ecd: ckks.NewEncoder(params)}
	r.enc, r.dec = rlwe.NewEncryptor(params, sk), rlwe.NewDecryptor(params, sk)
	c := &stdioConn{r: bufio.NewReaderSize(os.Stdin, 1<<20), w: bufio.NewWriterSize(os.Stdout, 1<<20)}
	if err := c.send(header{Kind: "opened", Fields: map[string]any{"profile": prof.ID, "profileDigest": prof.digest(),
		"insecureDevelopment": prof.Insecure, "keyEpoch": ep.Epoch, "keyEpochExpires": ep.Expires, "reused": reused,
		"keysDigest": ep.KeysDigest, "vocab": vocab, "context": context, "inputLevel": params.MaxLevel()}}); err != nil {
		return err
	}
	encryptions, decryptions := 0, 0
	for {
		h, blobs, err := c.recv()
		if err != nil {
			return err
		}
		switch h.Kind {
		case "keygen":
			// Each evaluation key is generated, sent to the host and dropped; the host forwards it to the evaluator.
			kgen := rlwe.NewKeyGenerator(params)
			var kd [][]byte
			rlk, err := kgen.GenRelinearizationKeyNew(sk).MarshalBinary()
			if err != nil {
				return err
			}
			kd = append(kd, []byte(digest(rlk)))
			if err := c.send(header{Kind: "key", Seq: 0}, rlk); err != nil {
				return err
			}
			rlk = nil
			for i, rot := range rotationSteps() {
				gk, err := kgen.GenGaloisKeyNew(params.GaloisElement(rot), sk).MarshalBinary()
				if err != nil {
					return err
				}
				kd = append(kd, []byte(digest(gk)))
				if err := c.send(header{Kind: "key", Seq: i + 1}, gk); err != nil {
					return err
				}
				runtime.GC()
			}
			if err := c.send(header{Kind: "keys-done", Fields: map[string]any{"keysDigest": digest(kd...)}}); err != nil {
				return err
			}
		case "commit-keys":
			// The evaluator acknowledged these keys for this epoch: later sessions may reuse them without upload.
			ep.KeysDigest, _ = h.Fields["keysDigest"].(string)
			if err := saveEpoch(o.stateDir, ep); err != nil {
				return err
			}
			if err := c.send(header{Kind: "committed"}); err != nil {
				return err
			}
		case "retire-epoch":
			// After a failed verification the host retires this key epoch: the stored secret key and epoch record are
			// deleted, so the next session makes a new key and uploads new evaluation keys.
			if o.stateDir != "" {
				os.Remove(filepath.Join(o.stateDir, "key-epoch.sk"))
				os.Remove(filepath.Join(o.stateDir, "key-epoch.json"))
			}
			ep.KeysDigest = ""
			if err := c.send(header{Kind: "retired", Fields: map[string]any{"keyEpoch": ep.Epoch}}); err != nil {
				return err
			}
		case "encrypt":
			raw, _ := h.Fields["tokens"].([]any)
			if len(raw) == 0 || len(raw) > 8*context {
				return fmt.Errorf("encrypt needs 1 to %d tokens", 8*context)
			}
			var out [][]byte
			for _, v := range raw {
				f, ok := v.(float64)
				if !ok || f != math.Trunc(f) || f < 0 || f >= vocab {
					return fmt.Errorf("token outside the vocabulary")
				}
				b, err := r.sealOneHot(int(f))
				if err != nil {
					return err
				}
				out = append(out, b)
				encryptions++
			}
			if err := c.send(header{Kind: "ciphertexts"}, out...); err != nil {
				return err
			}
		case "decrypt":
			if len(blobs) != 1 {
				return fmt.Errorf("decrypt takes one ciphertext")
			}
			ct := new(rlwe.Ciphertext)
			if err := ct.UnmarshalBinary(blobs[0]); err != nil || ct.Degree() != 1 || ct.LogDimensions.Cols != logSlots || !ct.IsNTT {
				// Shape failures answer without decrypting; the host decides whether the session survives.
				if err := c.send(header{Kind: "refused", Fields: map[string]any{"reason": "result does not have the pinned shape"}}); err != nil {
					return err
				}
				continue
			}
			decryptions++
			vals := make([]float64, slots)
			if err := r.ecd.Decode(r.dec.DecryptNew(ct), vals); err != nil {
				return err
			}
			out := make([]byte, 8*vocab)
			for i := 0; i < vocab; i++ {
				binary.LittleEndian.PutUint64(out[8*i:], math.Float64bits(vals[i]))
			}
			if err := c.send(header{Kind: "plaintext", Fields: map[string]any{"level": ct.Level()}}, out); err != nil {
				return err
			}
		case "close":
			return c.send(header{Kind: "closed", Fields: map[string]any{"encryptions": encryptions, "decryptions": decryptions}})
		}
	}
}
