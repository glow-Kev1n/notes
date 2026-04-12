# STL

## set与unordered_set都是可以遍历的

```cpp
unordered_set<int> S;
set<int> S;

for (int x : S) {
	...
	...
}
```

## Hash vs Compare (不同容器可用成员不同，pair)

---

## 📌 一、核心分类

### ✅ 1. 基于“比较（<）”的有序容器（Ordered Containers）

- `set`
    
- `map`
    
- `multiset`
    
- `multimap`
    

👉 **底层结构：红黑树（Balanced BST）**

👉 **核心依赖：比较函数（默认是 `<`）**

---

### ✅ 2. 基于“哈希（hash）”的无序容器（Unordered Containers）

- `unordered_set`
    
- `unordered_map`
    
- `unordered_multiset`
    
- `unordered_multimap`
    

👉 **底层结构：哈希表（Hash Table）**

👉 **核心依赖：**

- hash函数
    
- equality（=\=)

---

## 📌 二、对比总结（面试高频）

|特性|set / map|unordered_set / unordered_map|
|---|---|---|
|是否有序|✅ 有序|❌ 无序|
|底层结构|红黑树|哈希表|
|查找复杂度|O(log n)|O(1)（平均）|
|需要|`<` 比较|`hash` + `==`|
|是否支持 pair 默认使用|✅ 支持|❌ 不支持|

---

## 📌 三、pair 的支持情况

### ✅ 在 `set / map` 中

```cpp
std::set<std::pair<int,int>> s;
std::map<std::pair<int,int>, int> mp;
```

✔ 可以直接用  
✔ 因为 `pair` 已经定义了：

```cpp
operator<
```

👉 默认排序规则（字典序）：

```text
先比较 first
如果 first 相等，再比较 second
```

---

### ❌ 在 `unordered_set / unordered_map` 中

```cpp
std::unordered_set<std::pair<int,int>> s; // ❌ 报错
```

❗ 原因：

👉 标准库没有提供：

```cpp
std::hash<std::pair<...>>
```

---

### ✅ 解决方法：
  1、自定义 hash

```cpp
struct pair_hash {
    size_t operator()(const std::pair<int,int>& p) const {
        return hash<int>()(p.first) ^ (hash<int>()(p.second) << 1);
    }
};

std::unordered_set<std::pair<int,int>, pair_hash> s;
```

竞赛中常用这种⬇️
  2、对于数值不大的pair，可以压缩到一个long long或者int中
  ```cpp
  unordered_map<int, int> h;
  pair<int, int> p;
  //如果pair中元素 < 1e3
  int state = p.first << 10 | p.second;
  
  long long key = ((long long)first << 32) | (unsigned int)second;
  
  h[state] =  blabla;
  ```
---

## 📌 四、自定义比较函数（set / map）

### 🔧 示例：按 second 排序

```cpp
struct cmp {
    bool operator()(const std::pair<int,int>& a,
                    const std::pair<int,int>& b) const {
        return a.second < b.second;
    }
};

std::set<std::pair<int,int>, cmp> s;
```

---

### ⚠️ 经典坑（面试重点）

```cpp
return a.second < b.second;
```

❗ 问题：

👉 当 `a.second == b.second` 时  
➡️ 会被认为“相等”  
➡️ set 中只能存一个元素

---

### ✅ 正确写法（完整排序）

```cpp
return a.second < b.second ||
      (a.second == b.second && a.first < b.first);
```

---

## 📌 五、unordered 容器的要求

使用 unordered 容器时，必须满足：

### 1️⃣ hash 函数

```cpp
size_t hash(T)
```

### 2️⃣ equality 函数

```cpp
bool operator==(T a, T b)
```

---

👉 标准类型（int / string 等）已内置  
👉 自定义类型 / pair 需要手动实现 hash

---

## 📌 六、面试总结（必背）

### 🎯 核心一句话

> 有序容器（set/map）依赖 `<`，无序容器（unordered）依赖 `hash`

---

### 🎯 高频考点

- 为什么 `unordered_set<pair>` 报错？
    
- `set<pair>` 为什么可以直接用？
    
- 如何自定义 hash？
    
- comparator 写不严谨会导致什么问题？
    
- 红黑树 vs 哈希表的复杂度区别？
    

---

## 📌 七、扩展（加分项）

### 🚀 hash 冲突优化

简单写法：

```cpp
a ^ (b << 1)
```

更好方法：  
👉 使用 `boost::hash_combine`

---

### 🚀 使用场景总结

|场景|推荐容器|
|---|---|
|需要排序|set / map|
|只关心查找速度|unordered_set / unordered_map|
|key 是复杂结构|set/map 更简单（不用写 hash）|

---

# ✅ 最终记忆口诀

👉 **“要顺序用 set，要速度用 unordered”**  
👉 **“能比大小用 `<`，不能就写 hash”**

---