package main

// Framing: u32 header length | header JSON | blobs. The header lists blob sizes. Each message kind has its own
// size ceiling, checked before any blob byte is read. There is no message that asks the requester to decrypt,
// re-encrypt or evaluate anything: the requester treats any kind other than the one it awaits as an attack.

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"time"
)

const (
	maxHeader = 64 << 10
	// ceilings per kind (bytes of blobs in one message)
	maxKeyBytes    = 1 << 30
	maxStepBytes   = 8 * (64 << 20)
	maxResultBytes = 32 << 20
)

var kindCeiling = map[string]int64{
	"hello": 0, "manifest": 0, "key": maxKeyBytes, "ready": 0, "prefill": maxStepBytes, "decode": maxStepBytes,
	"result": maxResultBytes, "bye": 0,
}

type header struct {
	Kind     string         `json:"kind"`
	Protocol string         `json:"protocol"`
	Session  string         `json:"session"`
	Seq      int            `json:"seq"`
	Blobs    []int64        `json:"blobs"`
	Fields   map[string]any `json:"fields,omitempty"`
}

type conn struct {
	c              net.Conn
	sent, received int64
	deadline       time.Duration
}

func (c *conn) send(h header, blobs ...[]byte) error {
	h.Protocol = protocolID
	h.Blobs = nil
	for _, b := range blobs {
		h.Blobs = append(h.Blobs, int64(len(b)))
	}
	raw, err := json.Marshal(h)
	if err != nil {
		return err
	}
	c.c.SetWriteDeadline(time.Now().Add(c.deadline))
	var pre [4]byte
	binary.LittleEndian.PutUint32(pre[:], uint32(len(raw)))
	if _, err := c.c.Write(append(pre[:], raw...)); err != nil {
		return err
	}
	c.sent += int64(4 + len(raw))
	for _, b := range blobs {
		if _, err := c.c.Write(b); err != nil {
			return err
		}
		c.sent += int64(len(b))
	}
	return nil
}

// recv reads one message. want restricts the accepted kinds; anything else is an error before blobs are read.
func (c *conn) recv(want ...string) (header, [][]byte, error) {
	var h header
	c.c.SetReadDeadline(time.Now().Add(c.deadline))
	var pre [4]byte
	if _, err := io.ReadFull(c.c, pre[:]); err != nil {
		return h, nil, err
	}
	n := binary.LittleEndian.Uint32(pre[:])
	if n == 0 || n > maxHeader {
		return h, nil, fmt.Errorf("header length %d outside (0, %d]", n, maxHeader)
	}
	raw := make([]byte, n)
	if _, err := io.ReadFull(c.c, raw); err != nil {
		return h, nil, err
	}
	c.received += int64(4 + n)
	if err := json.Unmarshal(raw, &h); err != nil {
		return h, nil, fmt.Errorf("malformed header: %w", err)
	}
	ok := false
	for _, k := range want {
		ok = ok || h.Kind == k
	}
	if !ok {
		return h, nil, &refusal{Kind: h.Kind, Want: want}
	}
	if h.Protocol != protocolID {
		return h, nil, fmt.Errorf("protocol %q, expected %q", h.Protocol, protocolID)
	}
	var total int64
	for _, b := range h.Blobs {
		if b < 0 {
			return h, nil, fmt.Errorf("negative blob length")
		}
		total += b
	}
	if total > kindCeiling[h.Kind] {
		return h, nil, fmt.Errorf("%s carries %d bytes, above its %d ceiling", h.Kind, total, kindCeiling[h.Kind])
	}
	blobs := make([][]byte, len(h.Blobs))
	for i, b := range h.Blobs {
		blobs[i] = make([]byte, b)
		c.c.SetReadDeadline(time.Now().Add(c.deadline))
		if _, err := io.ReadFull(c.c, blobs[i]); err != nil {
			return h, nil, err
		}
		c.received += b
	}
	return h, blobs, nil
}

// refusal: the peer sent a message kind the receiver does not accept at this point (for the requester, any
// request to decrypt or otherwise act on peer-chosen ciphertext). The session ends; nothing is answered.
type refusal struct {
	Kind string
	Want []string
}

func (r *refusal) Error() string {
	return fmt.Sprintf("refused unsolicited %q message while waiting for %v; session ended without decrypting", r.Kind, r.Want)
}

func digest(b ...[]byte) string {
	h := sha256.New()
	for _, x := range b {
		var n [8]byte
		binary.LittleEndian.PutUint64(n[:], uint64(len(x)))
		h.Write(n[:])
		h.Write(x)
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}
