package main

import (
	"context"
	"errors"
	"sync"
)

/**

Задача реализовать функцию Do, которая будет получать данные из Producer и передавать их в Consumer. но  с условиями
1 - из продюссера может прийти пачка данных меньше MaxBatch - но в консюмер надо передавать данные пачками равными или чуть меньше чем в MaxBatch - те данные нужно  группировать
2 - после успешной отправки данных в консюмер надо подтвердить это коммитом в продюссере, коммиты должны идти строго в той же последовательности как и отправленные данные
реши задачу используя алгоритм потоки данных
*/

const MaxBatch = 9999

type Producer interface {
	Get() ([]any, int, error)
	Commit(commitId int) error
}

type Consumer interface {
	Push(data []any) error
}

// Структура для передачи данных в воркер
type dataCommit struct {
	data     []any
	commitId []int
}

func Do(p Producer, c Consumer) error {
	// Канал для передачи данных воркеру
	dataChan := make(chan dataCommit)
	// Канал для получения ошибок от воркера
	errChan := make(chan error, 1)
	defer close(errChan)

	ctx, cancel := context.WithCancel(context.Background())

	pushError := func(err error) {
		select {
		case errChan <- err:
		default:
		}
		cancel()
	}

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		defer close(dataChan)
		buffer := make([]any, 0, MaxBatch)
		commitsId := make([]int, 0)

		for {
			// Получаем данные из продюссера
			data, commitId, err := p.Get()
			if err != nil {
				if len(buffer) > 0 {
					// Если данные в буфере, отправляем их
					select {
					case dataChan <- dataCommit{data: buffer, commitId: commitsId}:
					case <-ctx.Done():
						return
					}
				}
				// Отправляем ошибку
				pushError(err)
				return
			}

			// Проверяем, не превышает ли размер данных максимальный размер пачки
			if len(data) > MaxBatch {
				// Если данные слишком велики, возвращаем ошибку
				pushError(errors.New("data size exceeds MaxBatch"))
				return
			}

			// Проверяем, помещаются ли новые данные в буфер
			if len(buffer)+len(data) <= MaxBatch {
				// Данные помещаются, добавляем их в буфер
				buffer = append(buffer, data...)
				commitsId = append(commitsId, commitId)
			} else {
				// Данные не помещаются, отправляем текущий буфер
				select {
				case dataChan <- dataCommit{data: buffer, commitId: commitsId}:
				case <-ctx.Done():
					return
				}
				// Начинаем новый буфер с новыми данными
				buffer = make([]any, 0, MaxBatch)
				buffer = append(buffer, data...)
				commitsId = append([]int{}, commitId)
			}
		}
	}()

	go func() {
		defer wg.Done()
		for {
			data, ok := <-dataChan
			if !ok {
				return
			}
			err := c.Push(data.data)
			if err != nil {
				pushError(err)
				return
			}
			for _, commitId := range data.commitId {
				if err := p.Commit(commitId); err != nil {
					pushError(err)
					return
				}
			}
			select {
			case <-ctx.Done():
				return
			default:
			}
		}
	}()

	err := <-errChan
	wg.Wait()
	return err
}
