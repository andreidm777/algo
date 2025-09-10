package main

func generate(current string, open, close, n int, result *[]string) {
	if len(current) == 2*n {
		*result = append(*result, current)
		return
	}
	if open < n {
		// пока есть открытые скобки, можно добавить еще одну
		generate(current+"(", open+1, close, n, result)
	}
	if close < open {
		generate(current+")", open, close+1, n, result)
	}
}

/*
*

	func main() {
		var n int
		_, _ = fmt.Fscanf(os.Stdin, "%d", &n)
		result := make([]string, 0)
		generate("", 0, 0, n, &result)
		for _, s := range result {
			fmt.Println(s)
		}
	}

type city struct {
	x int
	y int
}

func main() {
	var n, k, from, to int

	fmt.Scanf("%d", &n)

	cities := make([]city, 0, n)

	for n > 0 {
		cityTmp := city{}
		fmt.Scanf("%d %d", &cityTmp.x, &cityTmp.y)
		cities = append(cities, cityTmp)
		n--
	}

	fmt.Scanf("%d", &k)
	fmt.Scanf("%d %d", &from, &to)

	fmt.Println(-1)

}


package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

type City struct {
	x, y int
}

func main() {
	scanner := bufio.NewScanner(os.Stdin)

	// Читаем количество городов
	scanner.Scan()
	n, _ := strconv.Atoi(scanner.Text())

	// Читаем координаты городов
	cities := make([]City, n)
	for i := 0; i < n; i++ {
		scanner.Scan()
		coords := strings.Fields(scanner.Text())
		x, _ := strconv.Atoi(coords[0])
		y, _ := strconv.Atoi(coords[1])
		cities[i] = City{x, y}
	}

	// Читаем максимальное расстояние без дозаправки
	scanner.Scan()
	k, _ := strconv.Atoi(scanner.Text())

	// Читаем начальный и конечный город
	scanner.Scan()
	route := strings.Fields(scanner.Text())
	start, _ := strconv.Atoi(route[0])
	end, _ := strconv.Atoi(route[1])

	// Преобразуем в индексы (города нумеруются с 1)
	startIdx := start - 1
	endIdx := end - 1

	// Создаем граф смежности
	graph := make([][]int, n)
	for i := 0; i < n; i++ {
		graph[i] = make([]int, 0)
	}

	// Заполняем граф: добавляем ребра только если расстояние <= k
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			distance := calculateDistance(cities[i], cities[j])
			if distance <= k {
				graph[i] = append(graph[i], j)
				graph[j] = append(graph[j], i)
			}
		}
	}

	// Используем BFS для поиска кратчайшего пути
	visited := make([]bool, n)
	queue := []int{startIdx}
	visited[startIdx] = true
	steps := 0

	for len(queue) > 0 {
		size := len(queue)

		for i := 0; i < size; i++ {
			current := queue[0]
			queue = queue[1:]

			if current == endIdx {
				fmt.Println(steps)
				return
			}

			for _, neighbor := range graph[current] {
				if !visited[neighbor] {
					visited[neighbor] = true
					queue = append(queue, neighbor)
				}
			}
		}
		steps++
	}

	// Если не нашли путь
	fmt.Println(-1)
}

func calculateDistance(city1, city2 City) int {
	return int(math.Abs(float64(city1.x-city2.x)) + math.Abs(float64(city1.y-city2.y)))
}
*/

func neighbors(n []uint) uint {

}

/*
goods := []uint{5, 1, 3, 7}
byers := []uint{1, 2, 3, 4}

result := 2

есть массив потребностей пользователя и массив товаров - если число потребности не находится в товаре то потребитель покупает ближайший товар
а разница между потребностью и ценой товара - это недовольство пользователя - выдайте в результате сумму недовольств пользователя
напиши решение и сложность алгори
*/
/*
func main() {
	goods := []uint{5, 1, 3, 7}
	byers := []uint{1, 2, 3, 4}

	slices.Sort(goods)

	for i := 0; i < len(byers); i++ {

	}
}



каждая строка это день - в нем массив статистики юзеров - найдите победителей которые сделали больше всего шагов участвуя в каждом дне соревнований

statistics = [
	[{userId: 1, steps: 100},{userId: 2, steps: 100}],
	[{userId: 3, steps: 100},{userId: 2, steps: 100}],
]

ответ в виде

winer = {userIds:[2], steps: 200}
*/
