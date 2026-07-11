package main

import (
	"slices"
	"strconv"
	"testing"
)

func TestParseSize(t *testing.T) {
	for _, tt := range []struct {
		in   string
		want int
	}{
		{"0", 0},
		{"8192", 8192},
		{"1B", 1},
		{"8KiB", 8 << 10},
		{"8kib", 8 << 10},
		{"3MiB", 3 << 20},
		{"2GiB", 2 << 30},
	} {
		got, err := parseSize(tt.in)
		if err != nil {
			t.Errorf("parseSize(%q) error: %v", tt.in, err)
		} else if got != tt.want {
			t.Errorf("parseSize(%q) = %d, want %d", tt.in, got, tt.want)
		}
	}

	for _, in := range []string{"", "KiB", "1.5MiB", "8QiB", "abc"} {
		if _, err := parseSize(in); err == nil {
			t.Errorf("parseSize(%q) succeeded, want error", in)
		}
	}

	maxInt := int(^uint(0) >> 1)
	overflow := strconv.Itoa(maxInt/(1<<10)+1) + "KiB"
	if _, err := parseSize(overflow); err == nil {
		t.Errorf("parseSize(%q) succeeded, want overflow error", overflow)
	}
}

func TestParseSizes(t *testing.T) {
	got, err := parseSizes("1KiB, 64, 1KiB", "512:2048:512")
	if err != nil {
		t.Fatal(err)
	}
	want := []int{64, 512, 1024, 1536, 2048}
	if !slices.Equal(got, want) {
		t.Errorf("parseSizes = %v, want %v", got, want)
	}
}

func TestParseSizesErrors(t *testing.T) {
	for _, tt := range []struct{ list, sweep string }{
		{"0", ""},
		{"-5", ""},
		{"", "1:2"},
		{"", "0:100:10"},
		{"", "100:50:10"},
		{"", "50:100:0"},
		{"", "x:100:10"},
	} {
		if _, err := parseSizes(tt.list, tt.sweep); err == nil {
			t.Errorf("parseSizes(%q, %q) succeeded, want error", tt.list, tt.sweep)
		}
	}
}

func TestParseSizesSweepDoesNotOverflow(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	sweep := strconv.Itoa(maxInt-1) + ":" + strconv.Itoa(maxInt) + ":2"
	got, err := parseSizes("", sweep)
	if err != nil {
		t.Fatal(err)
	}
	want := []int{maxInt - 1}
	if !slices.Equal(got, want) {
		t.Errorf("parseSizes sweep = %v, want %v", got, want)
	}
}

func TestMedianMAD(t *testing.T) {
	odd := []float64{5, 1, 3}
	if got := median(odd); got != 3 {
		t.Errorf("median(odd) = %v, want 3", got)
	}
	even := []float64{4, 1, 3, 2}
	if got := median(even); got != 2.5 {
		t.Errorf("median(even) = %v, want 2.5", got)
	}
	// Deviations from 3: {2, 2, 0, 1, 1000} -> sorted {0, 1, 2, 2, 1000}.
	if got := mad([]float64{1, 5, 3, 4, 1003}, 3); got != 2 {
		t.Errorf("mad = %v, want 2", got)
	}
}

func TestFormatSize(t *testing.T) {
	for _, tt := range []struct {
		in   int
		want string
	}{
		{100, "100B"},
		{8192, "8KiB"},
		{8192 + 1, "8193B"},
		{1 << 20, "1MiB"},
		{3 << 30, "3GiB"},
	} {
		if got := formatSize(tt.in); got != tt.want {
			t.Errorf("formatSize(%d) = %q, want %q", tt.in, got, tt.want)
		}
	}
}
