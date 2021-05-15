
### 1.拓扑排序

场景：我们知道，一个完整的项目往往会包含很多代码源文件。编译器在编译整个项目的时候，需要按照依赖关系，依次编译每个源文件。比如，A.cpp 依赖 B.cpp，那在编译的时候，编译器需要先编译 B.cpp，才能编译 A.cpp。我们可以把源文件与源文件之间的依赖关系，抽象成一个有向图。每个源文件对应图中的一个顶点，源文件之间的依赖关系就是顶点之间的边。

```
from collections import deque
from itertools import filterfalse

class Graph:
    def __init__(self, num_vertices: int):
        self._num_vertices = num_vertices
        self._adjacency = [[] for _ in range(num_vertices)]
    
    def add_edge(self, s: int, t: int) -> None:
        self._adjacency[s].append(t)
        
    def tsort_by_kahn(self):
        """如果 s 需要先于 t 执行，那就添加一条 s 指向 t 的边
           找出一个入度为 0 的顶点，将其输出到拓扑排序的结果序列中（对应代码中就是把它打印出来），
           并且把这个顶点从图中删除（也就是把这个顶点可达的顶点的入度都减 1）。
           我们循环执行上面的过程，直到所有的顶点都被输出。
           最后输出的序列，就是满足局部依赖关系的拓扑排序。
        """        
        in_degree = [0] * self._num_vertices #计算入度
        for v in range(self._num_vertices):
            if len(self._adjacency[v]):
                for neighbour in self._adjacency[v]:
                    in_degree[neighbour] += 1
                    
        q = deque(filterfalse(lambda x: in_degree[x], range(self._num_vertices))) #入度为0的点的队列
        while q:
            v = q.popleft() #取出入度为0的点
            print(f"{v} -> ", end="")
            for neighbour in self._adjacency[v]:
                in_degree[neighbour] -= 1 #与其相连接的点入度减1
                if not in_degree[neighbour]: #如果入度为0，加入队列
                    q.append(neighbour)
        print("\b\b\b  ")         
        
if __name__ == "__main__":
    dag = Graph(4)
    dag.add_edge(1, 0)
    dag.add_edge(2, 1)
    dag.add_edge(1, 3)
    dag.tsort_by_kahn()
```

### 2.课程表

输入：numCourses = 2, prerequisites = [[1,0],[0,1]]
输出：false
解释：总共有 2 门课程。学习课程 1 之前，你需要先完成​课程 0 ；并且学习课程 0 之前，你还应先完成课程 1 。这是不可能的。

```
from collections import deque
from itertools import filterfalse

class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        adjacency = [[] for _ in range (numCourses)] 
        for i, j in prerequisites:
            adjacency[j].append(i)

        indegree = [0] * numCourses
        for i in range(numCourses):
            if len(adjacency[i]):
                for neighbour in adjacency[i]:
                    indegree[neighbour] += 1

        q = deque(filterfalse(lambda x : indegree[x], range(numCourses)))
        count = 0
        while q:
            v = q.popleft()
            count += 1
            for neighbour in adjacency[v]:
                indegree[neighbour] -= 1
                if not indegree[neighbour]:
                    q.append(neighbour)

        return count == numCourses
```

### 3.删除无效的括号

给你一个由若干括号和字母组成的字符串 s ，删除最小数量的无效括号，使得输入的字符串有效。

Ref：https://leetcode-cn.com/problems/remove-invalid-parentheses/solution/bfsjian-dan-er-you-xiang-xi-de-pythonjiang-jie-by-/

```
class Solution:
    def removeInvalidParentheses(self, s:str) -> List[str]:
        def isValid(s:str)->bool:
            cnt = 0
            for c in s:
                if c == "(": cnt += 1
                elif c == ")": cnt -= 1
                if cnt < 0: return False  # 只用中途cnt出现了负值，你就要终止循环，已经出现非法字符了
            return cnt == 0

        # BFS
        level = {s}  # 用set避免重复
        while True:
            valid = list(filter(isValid, level))  # 所有合法字符都筛选出来
            if valid: return valid # 如果当前valid是非空的，说明已经有合法的产生了
            # 下一层level
            next_level = set()
            for item in level:
                for i in range(len(item)):
                    if item[i] in "()":                     # 如果item[i]这个char是个括号就删了，如果不是括号就留着
                        next_level.add(item[:i]+item[i+1:])
            level = next_level
```