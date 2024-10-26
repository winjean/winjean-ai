from langchain_community.vectorstores import milvus


def main():
    set1 = {1, 2, 3}
    set2 = {3, 4, 5}
    union_set = set1.union(set2)
    update_set = set1.update(set2)
    print(set1)
    print(set2)
    print(union_set)
    print(update_set)


if __name__ == '__main__':
    main()
