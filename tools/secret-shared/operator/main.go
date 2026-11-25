// Command operator serves one share of one stage of a secret-shared split (Route 3).
//
// It holds the public weights of its stage's projections and computes W~ x_b (mod 2^64) on the share x_b it is sent.
// It never sees any other share and never evaluates a nonlinear function. Every received vector is a fresh, uniformly
// random (or SHAKE-256 pseudorandom) ring element, so on its own it learns nothing about the requester's content
// beyond sizes and timing.
//
//	operator --plan PLAN.json --model-dir DIR --stage S --share 0|1 --listen 127.0.0.1:7701 [--threads N]
//
// The session token (SS_TOKEN) must match the requester's hello. Wire format, little-endian: u32 magic "RSS1", u32 JSON
// length, JSON header, u64 payload length, payload.
package main

import (
	"bufio"
	"crypto/sha256"
	"crypto/sha3"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"os"
	"path/filepath"
	"time"
)

const magic = 0x31535352 // "RSS1"

type span struct {
	File   string `json:"file"`
	Offset int64  `json:"offset"`
	Size   int64  `json:"size"`
}
type planTensor struct {
	Name   string `json:"name"`
	Dtype  string `json:"dtype"`
	Shape  []int  `json:"shape"`
	Spans  []span `json:"spans"`
	Sha256 string `json:"sha256"`
}
type planGroup struct {
	ID      string       `json:"id"`
	In      int          `json:"in"`
	Tensors []planTensor `json:"tensors"`
}
type planStage struct {
	Index  int         `json:"index"`
	Groups []planGroup `json:"groups"`
}
type plan struct {
	Schema         string      `json:"schema"`
	ModelID        string      `json:"modelId"`
	ManifestSha256 string      `json:"manifestSha256"`
	Fx             uint        `json:"fx"`
	Fw             uint        `json:"fw"`
	Stages         []planStage `json:"stages"`
}

type group struct {
	id      string
	in, out int
	tensors []*Tensor
}

type stats struct {
	Rounds    int64 `json:"rounds"`
	Columns   int64 `json:"columns"`
	Macs      int64 `json:"macs"`
	ComputeNs int64 `json:"computeNs"`
	BytesIn   int64 `json:"bytesIn"`
	BytesOut  int64 `json:"bytesOut"`
}

func loadTensor(dir string, pt planTensor, fw uint) (*Tensor, int64, error) {
	var data []byte
	for _, s := range pt.Spans {
		f, err := os.Open(filepath.Join(dir, s.File))
		if err != nil {
			return nil, 0, err
		}
		buf := make([]byte, s.Size)
		_, err = f.ReadAt(buf, s.Offset)
		f.Close()
		if err != nil {
			return nil, 0, err
		}
		data = append(data, buf...)
	}
	sum := sha256.Sum256(data)
	if hex.EncodeToString(sum[:]) != pt.Sha256 {
		return nil, 0, fmt.Errorf("%s: digest does not match the plan", pt.Name)
	}
	rows, cols := pt.Shape[0], pt.Shape[1]
	t := &Tensor{Name: pt.Name, Rows: rows, Cols: cols}
	switch pt.Dtype {
	case "Q4_K_M", "Q4_K":
		if cols%256 != 0 || len(data) != rows*cols/256*q4kBlockBytes {
			return nil, 0, fmt.Errorf("%s: bad Q4_K size", pt.Name)
		}
		t.Q4K = data
	case "F16", "F32":
		t.Dense = make([]uint64, rows*cols)
		for i := range t.Dense {
			var v float64
			if pt.Dtype == "F16" {
				v = f16(binary.LittleEndian.Uint16(data[2*i:]))
			} else {
				v = float64(math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:])))
			}
			t.Dense[i] = fixed(v, fw)
		}
	default:
		return nil, 0, fmt.Errorf("%s: dtype %s", pt.Name, pt.Dtype)
	}
	return t, int64(len(data)), nil
}

func readMsg(r *bufio.Reader) (map[string]any, []byte, error) {
	var head [8]byte
	if _, err := io.ReadFull(r, head[:]); err != nil {
		return nil, nil, err
	}
	if binary.LittleEndian.Uint32(head[:4]) != magic {
		return nil, nil, errors.New("bad magic")
	}
	js := make([]byte, binary.LittleEndian.Uint32(head[4:]))
	if _, err := io.ReadFull(r, js); err != nil {
		return nil, nil, err
	}
	var plen [8]byte
	if _, err := io.ReadFull(r, plen[:]); err != nil {
		return nil, nil, err
	}
	payload := make([]byte, binary.LittleEndian.Uint64(plen[:]))
	if _, err := io.ReadFull(r, payload); err != nil {
		return nil, nil, err
	}
	var h map[string]any
	if err := json.Unmarshal(js, &h); err != nil {
		return nil, nil, err
	}
	return h, payload, nil
}

func writeMsg(w *bufio.Writer, h any, payload []byte) error {
	js, _ := json.Marshal(h)
	var head [8]byte
	binary.LittleEndian.PutUint32(head[:4], magic)
	binary.LittleEndian.PutUint32(head[4:], uint32(len(js)))
	w.Write(head[:])
	w.Write(js)
	var plen [8]byte
	binary.LittleEndian.PutUint64(plen[:], uint64(len(payload)))
	w.Write(plen[:])
	w.Write(payload)
	return w.Flush()
}

// prg expands the requester's seed for one round: SHAKE-256(seed || round as u64 LE), n*in*8 bytes.
func prg(seed []byte, round uint64, size int) []byte {
	h := sha3.NewSHAKE256()
	h.Write(seed)
	var ctr [8]byte
	binary.LittleEndian.PutUint64(ctr[:], round)
	h.Write(ctr[:])
	out := make([]byte, size)
	h.Read(out)
	return out
}

func main() {
	planFile := flag.String("plan", "", "plan JSON")
	dir := flag.String("model-dir", "", "directory holding the pinned shards")
	stageIndex := flag.Int("stage", -1, "stage index")
	share := flag.Int("share", -1, "share 0 (A) or 1 (B)")
	listen := flag.String("listen", "127.0.0.1:7701", "listen address")
	threads := flag.Int("threads", 0, "worker threads (0 = all CPUs)")
	flag.Parse()
	token := os.Getenv("SS_TOKEN")
	if *planFile == "" || *dir == "" || *stageIndex < 0 || (*share != 0 && *share != 1) || token == "" {
		log.Fatal("usage: operator --plan P --model-dir D --stage S --share 0|1 [--listen A] [--threads N], with SS_TOKEN set")
	}
	raw, err := os.ReadFile(*planFile)
	if err != nil {
		log.Fatal(err)
	}
	planSum := sha256.Sum256(raw)
	var p plan
	if err := json.Unmarshal(raw, &p); err != nil || p.Schema != "doppler.secret-shared-plan/v1" {
		log.Fatal("plan schema")
	}
	var st *planStage
	for i := range p.Stages {
		if p.Stages[i].Index == *stageIndex {
			st = &p.Stages[i]
		}
	}
	if st == nil {
		log.Fatal("stage not in plan")
	}
	started := time.Now()
	groups := map[string]*group{}
	var weightBytes int64
	for _, pg := range st.Groups {
		g := &group{id: pg.ID, in: pg.In}
		for _, pt := range pg.Tensors {
			t, n, err := loadTensor(*dir, pt, p.Fw)
			if err != nil {
				log.Fatal(err)
			}
			if t.Cols != pg.In {
				log.Fatalf("%s: width %d, group %d", t.Name, t.Cols, pg.In)
			}
			weightBytes += n
			g.tensors = append(g.tensors, t)
			g.out += t.Rows
		}
		groups[g.id] = g
	}
	loadMs := time.Since(started).Milliseconds()
	log.Printf("stage %d share %d: %d groups, %d weight bytes, loaded in %d ms, peak RSS %d", *stageIndex, *share, len(groups), weightBytes, loadMs, peakRSS())
	ln, err := net.Listen("tcp", *listen)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("READY %s\n", ln.Addr())
	for {
		conn, err := ln.Accept()
		if err != nil {
			log.Fatal(err)
		}
		if serve(conn, token, &p, *stageIndex, *share, groups, weightBytes, hex.EncodeToString(planSum[:]), *threads) {
			return
		}
	}
}

// serve handles one requester session; it returns true when the requester asked the operator to exit.
func serve(conn net.Conn, token string, p *plan, stage, share int, groups map[string]*group, weightBytes int64, planSha string, threads int) bool {
	defer conn.Close()
	if tc, ok := conn.(*net.TCPConn); ok {
		tc.SetNoDelay(true)
		tc.SetReadBuffer(8 << 20)
		tc.SetWriteBuffer(8 << 20)
	}
	r, w := bufio.NewReaderSize(conn, 1<<20), bufio.NewWriterSize(conn, 1<<20)
	var seed []byte
	var s stats
	authed := false
	for {
		h, payload, err := readMsg(r)
		if err != nil {
			return false
		}
		s.BytesIn += int64(len(payload))
		op, _ := h["op"].(string)
		if !authed && op != "hello" {
			writeMsg(w, map[string]any{"error": "hello first"}, nil)
			return false
		}
		switch op {
		case "hello":
			if h["token"] != token {
				writeMsg(w, map[string]any{"error": "token"}, nil)
				return false
			}
			authed = true
			s = stats{}
			info := []map[string]any{}
			for _, g := range groups {
				info = append(info, map[string]any{"id": g.id, "in": g.in, "out": g.out})
			}
			writeMsg(w, map[string]any{"stage": stage, "share": share, "planSha256": planSha, "modelId": p.ModelID,
				"fx": p.Fx, "fw": p.Fw, "groups": info, "weightBytes": weightBytes, "peakRss": peakRSS()}, nil)
		case "seed":
			b, err := hex.DecodeString(fmt.Sprint(h["seed"]))
			if err != nil || len(b) != 32 || share != 0 {
				writeMsg(w, map[string]any{"error": "seed is for share 0 only, 32 bytes"}, nil)
				return false
			}
			seed = b
			writeMsg(w, map[string]any{"ok": true}, nil)
		case "linear":
			g := groups[fmt.Sprint(h["group"])]
			n := int(h["n"].(float64))
			round := uint64(h["round"].(float64))
			if g == nil || n <= 0 {
				writeMsg(w, map[string]any{"error": "group"}, nil)
				return false
			}
			var in []byte
			if seeded, _ := h["seeded"].(bool); seeded {
				if seed == nil {
					writeMsg(w, map[string]any{"error": "no seed"}, nil)
					return false
				}
				in = prg(seed, round, n*g.in*8)
			} else {
				in = payload
			}
			if len(in) != n*g.in*8 {
				writeMsg(w, map[string]any{"error": "payload size"}, nil)
				return false
			}
			t0 := time.Now()
			x := make([]uint64, n*g.in)
			for i := range x {
				x[i] = binary.LittleEndian.Uint64(in[8*i:])
			}
			y := make([]uint64, n*g.out)
			off := 0
			for _, t := range g.tensors {
				t.MatVec(x, n, y, g.out, off, p.Fw, threads)
				off += t.Rows
			}
			out := make([]byte, 8*len(y))
			for i, v := range y {
				binary.LittleEndian.PutUint64(out[8*i:], v)
			}
			ns := time.Since(t0).Nanoseconds()
			s.Rounds++
			s.Columns += int64(n)
			s.Macs += int64(n * g.in * g.out)
			s.ComputeNs += ns
			s.BytesOut += int64(len(out))
			writeMsg(w, map[string]any{"round": round, "computeNs": ns}, out)
		case "stats":
			writeMsg(w, map[string]any{"stats": s, "peakRss": peakRSS(), "weightBytes": weightBytes}, nil)
		case "close":
			writeMsg(w, map[string]any{"stats": s, "peakRss": peakRSS()}, nil)
			return false
		case "exit":
			writeMsg(w, map[string]any{"stats": s, "peakRss": peakRSS()}, nil)
			return true
		default:
			writeMsg(w, map[string]any{"error": "op"}, nil)
			return false
		}
	}
}
