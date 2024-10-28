from multiprocessing import Process, Queue, Value, Array, Lock, Pool


def worker(queue, description):
    queue.put(description)


def invoke_queue():
    queue = Queue()
    process = Process(target=worker, args=(queue, 'Hello from the child process'))
    process.start()
    print(queue.get())
    process.join()


def update_value(num, arr):
    num.value = 3.14
    for i in range(len(arr)):
        arr[i] = -arr[i]


def invoke_value():
    num = Value('d', 0.0)
    arr = Array('i', [1, 2, 3, 4, 5])

    p = Process(target=update_value, args=(num, arr))
    p.start()
    p.join()

    print(num.value)
    print(arr[:])


if __name__ == '__main__':
    # invoke_queue()
    invoke_value()

