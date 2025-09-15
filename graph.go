package main

import (
	"fmt"
	"math"
)

type city struct {
	x int
	y int
}

func calculateDistance(city1, city2 city) int {
	return int(math.Abs(float64(city1.x-city2.x)) + math.Abs(float64(city1.y-city2.y)))
}

func twoSum(nums []int, target int) []int {
	sum := make(map[int]int)

	for i, v := range nums {
		ans := target - v
		if idx, ok := sum[ans]; ok {
			return []int{idx, i}
		} else {
			sum[ans] = i
		}
	}

	return []int{}
}

func main() {
	/*
		var n, k, from, to int

		fmt.Scanf("%d", &n)

		cities := make([]city, 0, n)

		y := n
		for y > 0 {
			cityTmp := city{}
			fmt.Scanf("%d %d", &cityTmp.x, &cityTmp.y)
			cities = append(cities, cityTmp)
			y--
		}

		fmt.Scanf("%d", &k)
		fmt.Scanf("%d %d", &from, &to)

		start := from - 1
		end := to - 1

		graph := make([][]int, n)

		for i := 0; i < n; i++ {
			graph[i] = make([]int, 0)
		}

		fmt.Println(graph)

		for i := 0; i < len(cities); i++ {
			for j := i + 1; j < len(cities); j++ {
				dist := calculateDistance(cities[i], cities[j])
				if dist <= k {
					graph[i] = append(graph[i], j)
					graph[j] = append(graph[j], i)
				}
			}
		}

		queue := []int{start}
		visited := make(map[int]bool)
		step := 0

		fmt.Println(graph)

		for len(queue) > 0 {
			stepLen := len(queue)
			for i := 0; i < stepLen; i++ {
				cursor := queue[0]
				queue = queue[1:]

				if !visited[cursor] {
					visited[cursor] = true
				}

				if cursor == end {
					fmt.Println(step)
					return
				}

				for _, next := range graph[cursor] {
					if !visited[next] {
						queue = append(queue, next)
					}
				}
			}
			step++
		}

		fmt.Println(-1)

		//b := B(&C{})
	*/
	var b *C

	fmt.Println(b.Get())

	arr := [10]int{}

	arr2 := [10]int{}

	arr[2] = 10
	arr2[2] = 9

	arr2[2]++

	fmt.Println(arr2 == arr)

}

type B interface {
	Get() int
}

type C struct {
}

func (c *C) Get() int {
	return 1
}
