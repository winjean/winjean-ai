from langchain_core.runnables.base import RunnableSequence
from langchain_core.runnables import RunnableLambda

# 定义任务
def task1(input_data):
  print("Task 1:", input_data)
  return input_data + " processed by Task 1"

def task2(input_data):
  print("Task 2:", input_data)
  return input_data + " processed by Task 2"

def task3(input_data):
  print("Task 3:", input_data)
  return input_data + " processed by Task 3"

def task4(input_data):
  print("Task 4:", input_data)
  return input_data + " processed by Task 4"

runnable_1 = RunnableLambda(task1)
runnable_2 = RunnableLambda(task2)
runnable_3 = RunnableLambda(task3)
runnable_4 = RunnableLambda(task4)

middle=[runnable_2,runnable_3]

# 创建序列
sequence = RunnableSequence(first=runnable_1, middle=middle, last=runnable_4)

# 执行序列
result = sequence.invoke("Initial data")
print("Final result:", result)

if __name__ == '__main__':
    pass