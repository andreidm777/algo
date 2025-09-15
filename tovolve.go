package main

import (
	"container/heap"
	"strings"
	"testing"
)

type alpha struct {
	c   rune
	idx int
}

type MyHeap []alpha

func (h MyHeap) Len() int {
	return len(h)
}

func (h MyHeap) Less(i, j int) bool {
	return (h[i].c < h[j].c)
}

func (h MyHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
	x := h[i].idx
	h[i].idx = h[j].idx
	h[j].idx = x
}

func (h *MyHeap) Push(v any) {
	*h = append(*h, v.(alpha))
}

func (h *MyHeap) Pop() any {
	old := *h
	x := old[len(old)-1]
	*h = old[:len(old)-1]
	return x
}

func isVowel(a rune) bool {
	all := "qeyuioajQEYUIOAJ"
	return strings.ContainsRune(all, a)
}

func sortVowels(s string) string {
	h := make(MyHeap, 0)
	for i, v := range s {
		if isVowel(v) {
			heap.Push(&h, alpha{v, i})
		}
	}

	res := []rune(s)

	for len(h) > 0 {
		v := heap.Pop(&h).(alpha)
		res[v.idx] = v.c
	}

	return string(res)
}

func TestVolvels(t *testing.T) {
	for _, test := range []struct {
		s    string
		want string
	}{
		{"lEetcOde", "lEOtcede"},
		{"lYmpH", "lYmpH"},
	} {
		if got := sortVowels(test.s); got != test.want {
			t.Errorf("sortVowels(%q) = %q, want %q", test.s, got, test.want)
		}
	}
}
