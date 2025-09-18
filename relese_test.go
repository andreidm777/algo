package main

import (
	"errors"
	"testing"
)

// MockProducer - мock для Producer
type MockProducer struct {
	data            [][]any
	commitIds       []int
	getCallCount    int
	getError        error
	commitCallCount int
	commitError     error
}

func (mp *MockProducer) Get() ([]any, int, error) {
	if mp.getError != nil {
		return nil, 0, mp.getError
	}

	if mp.getCallCount >= len(mp.data) {
		// Сигнализируем о завершении
		return nil, 0, errors.New("no more data")
	}

	data := mp.data[mp.getCallCount]
	commitId := mp.commitIds[mp.getCallCount]
	mp.getCallCount++
	return data, commitId, nil
}

func (mp *MockProducer) Commit(commitId int) error {
	if mp.commitError != nil {
		return mp.commitError
	}
	mp.commitCallCount++
	return nil
}

// MockConsumer - мock для Consumer
type MockConsumer struct {
	pushCallCount int
	pushError     error
	receivedData  [][]any
}

func (mc *MockConsumer) Push(data []any) error {
	if mc.pushError != nil {
		return mc.pushError
	}
	mc.pushCallCount++
	mc.receivedData = append(mc.receivedData, data)
	return nil
}

func TestDo_NormalOperation(t *testing.T) {
	// Тест на нормальную работу
	producer := &MockProducer{
		data: [][]any{
			{1, 2, 3},
			{4, 5},
			{6, 7, 8, 9},
		},
		commitIds: []int{1, 2, 3},
	}

	consumer := &MockConsumer{}

	err := Do(producer, consumer)

	// Ожидаем ошибку "no more data" как сигнал завершения
	if err == nil || err.Error() != "no more data" {
		t.Errorf("Expected 'no more data' error, got %v", err)
	}

	// Проверяем, что данные были отправлены
	if len(consumer.receivedData) == 0 {
		t.Error("Expected data to be pushed to consumer")
	}

	// Проверяем, что коммиты были сделаны
	if producer.commitCallCount != 3 {
		t.Errorf("Expected 3 commits, got %d", producer.commitCallCount)
	}
}

func TestDo_DataExceedsMaxBatch(t *testing.T) {
	// Тест на случай, когда данные превышают MaxBatch
	producer := &MockProducer{
		data: [][]any{
			make([]any, MaxBatch+1), // Данные больше MaxBatch
		},
		commitIds: []int{1},
	}

	consumer := &MockConsumer{}

	err := Do(producer, consumer)

	// Ожидаем ошибку "data size exceeds MaxBatch"
	if err == nil || err.Error() != "data size exceeds MaxBatch" {
		t.Errorf("Expected 'data size exceeds MaxBatch' error, got %v", err)
	}

	// Проверяем, что данные не были отправлены
	if len(consumer.receivedData) != 0 {
		t.Error("Expected no data to be pushed to consumer")
	}

	// Проверяем, что коммиты не были сделаны
	if producer.commitCallCount != 0 {
		t.Errorf("Expected 0 commits, got %d", producer.commitCallCount)
	}
}

func TestDo_ProducerGetError(t *testing.T) {
	// Тест на обработку ошибки от Producer.Get()
	producer := &MockProducer{
		getError: errors.New("producer get error"),
	}

	consumer := &MockConsumer{}

	err := Do(producer, consumer)

	// Ожидаем ошибку от Producer.Get()
	if err == nil || err.Error() != "producer get error" {
		t.Errorf("Expected 'producer get error', got %v", err)
	}

	// Проверяем, что данные не были отправлены
	if len(consumer.receivedData) != 0 {
		t.Error("Expected no data to be pushed to consumer")
	}

	// Проверяем, что коммиты не были сделаны
	if producer.commitCallCount != 0 {
		t.Errorf("Expected 0 commits, got %d", producer.commitCallCount)
	}
}

func TestDo_ConsumerPushError(t *testing.T) {
	// Тест на обработку ошибки от Consumer.Push()
	producer := &MockProducer{
		data: [][]any{
			make([]any, 9999),
			{4, 5, 6},
		},
		commitIds: []int{1, 2},
	}

	consumer := &MockConsumer{
		pushError: errors.New("consumer push error"),
	}

	err := Do(producer, consumer)

	// Ожидаем ошибку от Consumer.Push()
	if err == nil || err.Error() != "consumer push error" {
		t.Errorf("Expected 'consumer push error', got %v", err)
	}

	// Проверяем, что данные не были отправлены (или были отправлены, но обработка прервана)
	// В данном случае, данные отправляются, но обработка прерывается на первой ошибке
	// Так что receivedData может быть 1 или 0, в зависимости от точного момента прерывания
	// Но важно, что ошибка возвращается
}

func TestDo_ProducerCommitError(t *testing.T) {
	// Тест на обработку ошибки от Producer.Commit()
	producer := &MockProducer{
		data: [][]any{
			make([]any, 9999),
			{2, 3},
		},
		commitIds:   []int{1, 2},
		commitError: errors.New("producer commit error"),
	}

	consumer := &MockConsumer{}

	err := Do(producer, consumer)

	// Ожидаем ошибку от Producer.Commit()
	if err == nil || err.Error() != "producer commit error" {
		t.Errorf("Expected 'producer commit error', got %v", err)
	}

	// Проверяем, что данные были отправлены
	if len(consumer.receivedData) == 0 {
		t.Error("Expected data to be pushed to consumer")
	}
}
