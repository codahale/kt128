// Command kt128bench measures KT128 hashing throughput across a set of input
// sizes with enough samples per size to report robust statistics. Sweeping
// sizes at fine granularity exposes discontinuities in the SIMD leaf
// scheduling (8 KiB leaf boundaries, lane-batch remainders).
package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/codahale/kt128"
)

func main() {
	var (
		sizesFlag  = flag.String("sizes", "", "comma-separated input sizes (e.g. 1KiB,8KiB,1MiB)")
		sweepFlag  = flag.String("sweep", "", "linear size sweep as min:max:step (e.g. 8KiB:512KiB:4KiB)")
		samples    = flag.Int("samples", 20, "timed samples per size")
		minTime    = flag.Duration("mintime", 50*time.Millisecond, "minimum wall time per sample")
		warmup     = flag.Int("warmup", 3, "discarded warmup samples per size")
		chunk      = flag.String("chunk", "0", "if >0, feed input via repeated Writes of this size instead of one-shot")
		formatFlag = flag.String("format", "table", "output format: table or csv")
	)
	flag.Parse()

	sizes, err := parseSizes(*sizesFlag, *sweepFlag)
	if err != nil {
		fmt.Fprintln(os.Stderr, "kt128bench:", err)
		os.Exit(2)
	}
	if len(sizes) == 0 {
		fmt.Fprintln(os.Stderr, "kt128bench: no sizes given; use -sizes and/or -sweep")
		os.Exit(2)
	}
	chunkSize, err := parseSize(*chunk)
	if err != nil {
		fmt.Fprintln(os.Stderr, "kt128bench: -chunk:", err)
		os.Exit(2)
	}
	if *samples < 1 || *warmup < 0 || *minTime <= 0 {
		fmt.Fprintln(os.Stderr, "kt128bench: -samples must be >= 1, -warmup >= 0, -mintime > 0")
		os.Exit(2)
	}
	var emit func(r result)
	switch *formatFlag {
	case "table":
		fmt.Printf("%12s %9s %8s %12s %12s %12s %8s %8s\n",
			"size", "iters", "samples", "median", "min", "max", "±mad", "GB/s")
		emit = emitTable
	case "csv":
		fmt.Println("size,iters,samples,median_ns,min_ns,max_ns,mad_pct,gbps")
		emit = emitCSV
	default:
		fmt.Fprintf(os.Stderr, "kt128bench: unknown format %q\n", *formatFlag)
		os.Exit(2)
	}

	// One pattern-filled buffer at the max size, re-sliced per size. In chunked
	// mode only the chunk window is needed.
	bufSize := sizes[len(sizes)-1]
	if chunkSize > 0 && chunkSize < bufSize {
		bufSize = chunkSize
	}
	msg := ptn(bufSize)

	runtime.LockOSThread()
	for _, size := range sizes {
		emit(measure(msg, size, chunkSize, *samples, *warmup, *minTime))
	}
}

// ptn fills a buffer with the RFC 9861 test pattern (cyclic 0x00..0xFA).
func ptn(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(i % 251)
	}
	return b
}

// hash runs one iteration: absorb size bytes and read 32 bytes to finalize.
// The Read is not part of the workload being measured but forces the final
// node to be chained and squeezed, so the full input is actually hashed.
func hash(msg []byte, size, chunkSize int, out *[32]byte) {
	h := kt128.New(nil)
	if chunkSize <= 0 {
		_, _ = h.Write(msg[:size])
	} else {
		for wrote := 0; wrote < size; wrote += chunkSize {
			_, _ = h.Write(msg[:min(chunkSize, size-wrote)])
		}
	}
	_, _ = h.Read(out[:])
}

type result struct {
	size, iters, samples           int
	medianNS, minNS, maxNS, madPct float64
}

func measure(msg []byte, size, chunkSize, samples, warmup int, minTime time.Duration) result {
	var out [32]byte
	runtime.GC()

	// Calibrate the per-sample iteration count so one batch runs >= minTime,
	// growing the trial count geometrically like testing.B does.
	iters := 1
	for {
		start := time.Now()
		for range iters {
			hash(msg, size, chunkSize, &out)
		}
		elapsed := time.Since(start)
		if elapsed >= minTime {
			break
		}
		// Predict from the observed rate, with headroom; at least double.
		next := int(float64(iters) * 1.2 * float64(minTime) / float64(max(elapsed, time.Microsecond)))
		iters = max(next, iters*2)
	}

	nsPerOp := make([]float64, 0, samples)
	for s := 0; s < warmup+samples; s++ {
		start := time.Now()
		for range iters {
			hash(msg, size, chunkSize, &out)
		}
		elapsed := time.Since(start)
		if s >= warmup {
			nsPerOp = append(nsPerOp, float64(elapsed.Nanoseconds())/float64(iters))
		}
	}

	med := median(nsPerOp)
	return result{
		size:     size,
		iters:    iters,
		samples:  samples,
		medianNS: med,
		minNS:    slices.Min(nsPerOp),
		maxNS:    slices.Max(nsPerOp),
		madPct:   mad(nsPerOp, med) / med * 100,
	}
}

func (r result) gbps() float64 {
	return float64(r.size) / r.medianNS // bytes/ns == GB/s
}

func emitTable(r result) {
	fmt.Printf("%12s %9d %8d %12s %12s %12s %7.2f%% %8.3f\n",
		formatSize(r.size), r.iters, r.samples,
		formatNS(r.medianNS), formatNS(r.minNS), formatNS(r.maxNS),
		r.madPct, r.gbps())
}

func emitCSV(r result) {
	fmt.Printf("%d,%d,%d,%.1f,%.1f,%.1f,%.3f,%.4f\n",
		r.size, r.iters, r.samples, r.medianNS, r.minNS, r.maxNS, r.madPct, r.gbps())
}

// median returns the median of xs without modifying it.
func median(xs []float64) float64 {
	s := slices.Clone(xs)
	sort.Float64s(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

// mad returns the median absolute deviation of xs from center.
func mad(xs []float64, center float64) float64 {
	devs := make([]float64, len(xs))
	for i, x := range xs {
		if x >= center {
			devs[i] = x - center
		} else {
			devs[i] = center - x
		}
	}
	return median(devs)
}

// parseSizes merges the -sizes list and the -sweep range into a sorted,
// deduplicated slice of sizes in bytes.
func parseSizes(list, sweep string) ([]int, error) {
	var sizes []int
	if list != "" {
		for f := range strings.SplitSeq(list, ",") {
			n, err := parseSize(strings.TrimSpace(f))
			if err != nil {
				return nil, fmt.Errorf("-sizes: %w", err)
			}
			if n <= 0 {
				return nil, fmt.Errorf("-sizes: size must be positive, got %q", f)
			}
			sizes = append(sizes, n)
		}
	}
	if sweep != "" {
		parts := strings.Split(sweep, ":")
		if len(parts) != 3 {
			return nil, fmt.Errorf("-sweep: want min:max:step, got %q", sweep)
		}
		var v [3]int
		for i, p := range parts {
			n, err := parseSize(strings.TrimSpace(p))
			if err != nil {
				return nil, fmt.Errorf("-sweep: %w", err)
			}
			v[i] = n
		}
		lo, hi, step := v[0], v[1], v[2]
		if lo <= 0 || hi < lo || step <= 0 {
			return nil, fmt.Errorf("-sweep: want 0 < min <= max and step > 0, got %q", sweep)
		}
		for n := lo; n <= hi; n += step {
			sizes = append(sizes, n)
		}
	}
	slices.Sort(sizes)
	return slices.Compact(sizes), nil
}

var units = []struct {
	suffix string
	mult   int
}{
	{"GiB", 1 << 30},
	{"MiB", 1 << 20},
	{"KiB", 1 << 10},
	{"B", 1},
}

// parseSize parses a size like "8192", "8KiB", or "1MiB" (case-insensitive).
func parseSize(s string) (int, error) {
	num, mult := s, 1
	for _, u := range units {
		if len(s) > len(u.suffix) && strings.EqualFold(s[len(s)-len(u.suffix):], u.suffix) {
			num, mult = s[:len(s)-len(u.suffix)], u.mult
			break
		}
	}
	n, err := strconv.Atoi(strings.TrimSpace(num))
	if err != nil {
		return 0, fmt.Errorf("invalid size %q", s)
	}
	return n * mult, nil
}

func formatSize(n int) string {
	for _, u := range units[:3] {
		if n >= u.mult && n%u.mult == 0 {
			return strconv.Itoa(n/u.mult) + u.suffix
		}
	}
	return strconv.Itoa(n) + "B"
}

func formatNS(ns float64) string {
	return time.Duration(ns).Round(10 * time.Nanosecond).String()
}
