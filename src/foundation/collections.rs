//! OCCT-style collection classes
//!
//! Provides:
//! - `Array1<T>` (TColStd_Array1) - 1D dynamic array with optional 1-based indexing
//! - `List<T>` (TColStd_List) - doubly-linked list
//! - `Map<K, V>` (TColStd_Map) - hash map wrapper
//! - `Set<T>` (TColStd_Set) - hash set wrapper

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;

/// A 1D array with optional 1-based indexing (OCCT TColStd_Array1 equivalent)
///
/// By default uses 0-based indexing for Rust compatibility.
/// Can be created with 1-based indexing by using `Array1::new_1based()`.
#[derive(Debug, Clone)]
pub struct Array1<T: Clone> {
    data: Vec<T>,
    lower: usize,  // Lower index (0 for 0-based, 1 for 1-based)
}

impl<T: Clone> Array1<T> {
    /// Create a new array with 0-based indexing and initial capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            lower: 0,
        }
    }

    /// Create a new 1-based indexed array (OCCT style)
    ///
    /// This array uses 1-based indexing like OCCT's TColStd_Array1.
    /// Indices from 1 to length are valid.
    pub fn new_1based(length: usize) -> Self {
        Self {
            data: Vec::with_capacity(length),
            lower: 1,
        }
    }

    /// Create array from a vector of elements
    pub fn from_vec(vec: Vec<T>) -> Self {
        let len = vec.len();
        let mut arr = Self::new(len);
        arr.data = vec;
        arr
    }

    /// Create 1-based array from a vector of elements
    pub fn from_vec_1based(vec: Vec<T>) -> Self {
        let len = vec.len();
        let mut arr = Self::new_1based(len);
        arr.data = vec;
        arr
    }

    /// Get the lower index bound
    pub fn lower_bound(&self) -> usize {
        self.lower
    }

    /// Get the number of elements in the array (OCCT Length())
    pub fn length(&self) -> usize {
        self.data.len()
    }

    /// Get the upper index bound (for 1-based arrays, this is length; for 0-based, length - 1)
    pub fn upper_bound(&self) -> usize {
        if self.lower == 0 {
            self.data.len().saturating_sub(1)
        } else {
            self.data.len()
        }
    }

    /// Get element at index (OCCT Value())
    ///
    /// For 0-based arrays, index must be 0 to length-1.
    /// For 1-based arrays, index must be 1 to length.
    pub fn value(&self, index: usize) -> Option<T> {
        if index < self.lower {
            return None;
        }
        let actual_index = index - self.lower;
        self.data.get(actual_index).cloned()
    }

    /// Set element at index (OCCT SetValue())
    ///
    /// For 0-based arrays, index must be 0 to length-1.
    /// For 1-based arrays, index must be 1 to length.
    pub fn set_value(&mut self, index: usize, value: T) -> Result<(), String> {
        if index < self.lower {
            return Err(format!("Index {} out of bounds (lower bound is {})", index, self.lower));
        }
        let actual_index = index - self.lower;
        if actual_index < self.data.len() {
            self.data[actual_index] = value;
            Ok(())
        } else {
            Err(format!("Index {} out of bounds", index))
        }
    }

    /// Append element to the end
    pub fn append(&mut self, value: T) {
        self.data.push(value);
    }

    /// Insert element at specific index
    pub fn insert(&mut self, index: usize, value: T) -> Result<(), String> {
        if index < self.lower {
            return Err(format!("Index {} out of bounds (lower bound is {})", index, self.lower));
        }
        let actual_index = index - self.lower;
        if actual_index <= self.data.len() {
            self.data.insert(actual_index, value);
            Ok(())
        } else {
            Err(format!("Index {} out of bounds for insertion", index))
        }
    }

    /// Remove element at index
    pub fn remove(&mut self, index: usize) -> Result<T, String> {
        if index < self.lower {
            return Err(format!("Index {} out of bounds (lower bound is {})", index, self.lower));
        }
        let actual_index = index - self.lower;
        if actual_index < self.data.len() {
            Ok(self.data.remove(actual_index))
        } else {
            Err(format!("Index {} out of bounds", index))
        }
    }

    /// Check if array is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Clear all elements
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Get underlying slice
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Get mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Iterate over elements
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }

    /// Mutable iterator over elements
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }

    /// Convert to owned vector
    pub fn to_vec(&self) -> Vec<T> {
        self.data.clone()
    }
}

/// A doubly-linked list (OCCT TColStd_List equivalent)
///
/// Provides efficient insertion and removal at any position.
#[derive(Debug, Clone)]
pub struct List<T: Clone> {
    data: VecDeque<T>,
}

impl<T: Clone> List<T> {
    /// Create a new empty list
    pub fn new() -> Self {
        Self {
            data: VecDeque::new(),
        }
    }

    /// Create a list with a specific capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: VecDeque::with_capacity(capacity),
        }
    }

    /// Get the number of elements (OCCT Length())
    pub fn length(&self) -> usize {
        self.data.len()
    }

    /// Get element at index (OCCT Value())
    pub fn value(&self, index: usize) -> Option<T> {
        self.data.get(index).cloned()
    }

    /// Set element at index (OCCT SetValue())
    pub fn set_value(&mut self, index: usize, value: T) -> Result<(), String> {
        if index < self.data.len() {
            self.data[index] = value;
            Ok(())
        } else {
            Err(format!("Index {} out of bounds", index))
        }
    }

    /// Append element to the end
    pub fn append(&mut self, value: T) {
        self.data.push_back(value);
    }

    /// Prepend element to the front
    pub fn prepend(&mut self, value: T) {
        self.data.push_front(value);
    }

    /// Insert element at specific index
    pub fn insert(&mut self, index: usize, value: T) -> Result<(), String> {
        if index <= self.data.len() {
            self.data.insert(index, value);
            Ok(())
        } else {
            Err(format!("Index {} out of bounds for insertion", index))
        }
    }

    /// Remove element at index
    pub fn remove(&mut self, index: usize) -> Result<T, String> {
        if index < self.data.len() {
            self.data.remove(index).ok_or_else(|| format!("Failed to remove at index {}", index))
        } else {
            Err(format!("Index {} out of bounds", index))
        }
    }

    /// Remove the first element
    pub fn remove_first(&mut self) -> Option<T> {
        self.data.pop_front()
    }

    /// Remove the last element
    pub fn remove_last(&mut self) -> Option<T> {
        self.data.pop_back()
    }

    /// Get the first element
    pub fn first(&self) -> Option<T> {
        self.data.front().cloned()
    }

    /// Get the last element
    pub fn last(&self) -> Option<T> {
        self.data.back().cloned()
    }

    /// Check if list is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Clear all elements
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Iterate over elements
    pub fn iter(&self) -> std::collections::vec_deque::Iter<'_, T> {
        self.data.iter()
    }

    /// Mutable iterator over elements
    pub fn iter_mut(&mut self) -> std::collections::vec_deque::IterMut<'_, T> {
        self.data.iter_mut()
    }

    /// Convert to vector
    pub fn to_vec(&self) -> Vec<T> {
        self.data.iter().cloned().collect()
    }
}

impl<T: Clone> Default for List<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// A hash map wrapper (OCCT TColStd_Map equivalent)
///
/// Maps keys to values using a hash table.
#[derive(Debug, Clone)]
pub struct Map<K: Hash + Eq + Clone, V: Clone> {
    data: HashMap<K, V>,
}

impl<K: Hash + Eq + Clone, V: Clone> Map<K, V> {
    /// Create a new empty map
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Create a map with a specific capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: HashMap::with_capacity(capacity),
        }
    }

    /// Get the number of key-value pairs (OCCT Length())
    pub fn length(&self) -> usize {
        self.data.len()
    }

    /// Get value for key (OCCT Find() or Value())
    pub fn value(&self, key: &K) -> Option<V> {
        self.data.get(key).cloned()
    }

    /// Insert or update a key-value pair
    pub fn insert(&mut self, key: K, value: V) {
        self.data.insert(key, value);
    }

    /// Set value for key (OCCT Bind())
    pub fn bind(&mut self, key: K, value: V) -> Option<V> {
        self.data.insert(key, value)
    }

    /// Remove a key-value pair
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.data.remove(key)
    }

    /// Check if map contains key
    pub fn contains(&self, key: &K) -> bool {
        self.data.contains_key(key)
    }

    /// Check if map is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Get all keys
    pub fn keys(&self) -> Vec<K> {
        self.data.keys().cloned().collect()
    }

    /// Get all values
    pub fn values(&self) -> Vec<V> {
        self.data.values().cloned().collect()
    }

    /// Iterate over key-value pairs
    pub fn iter(&self) -> std::collections::hash_map::Iter<'_, K, V> {
        self.data.iter()
    }

    /// Mutable iterator over key-value pairs
    pub fn iter_mut(&mut self) -> std::collections::hash_map::IterMut<'_, K, V> {
        self.data.iter_mut()
    }
}

impl<K: Hash + Eq + Clone, V: Clone> Default for Map<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

/// A hash set wrapper (OCCT TColStd_Set equivalent)
///
/// A collection of unique values using a hash table.
#[derive(Debug, Clone)]
pub struct Set<T: Hash + Eq + Clone> {
    data: HashSet<T>,
}

impl<T: Hash + Eq + Clone> Set<T> {
    /// Create a new empty set
    pub fn new() -> Self {
        Self {
            data: HashSet::new(),
        }
    }

    /// Create a set with a specific capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: HashSet::with_capacity(capacity),
        }
    }

    /// Get the number of elements (OCCT Length())
    pub fn length(&self) -> usize {
        self.data.len()
    }

    /// Add an element to the set
    pub fn add(&mut self, value: T) -> bool {
        self.data.insert(value)
    }

    /// Insert an element (alias for add, OCCT Bind())
    pub fn insert(&mut self, value: T) -> bool {
        self.data.insert(value)
    }

    /// Remove an element
    pub fn remove(&mut self, value: &T) -> bool {
        self.data.remove(value)
    }

    /// Check if set contains an element
    pub fn contains(&self, value: &T) -> bool {
        self.data.contains(value)
    }

    /// Check if set is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Clear all elements
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Get all elements as a vector
    pub fn to_vec(&self) -> Vec<T> {
        self.data.iter().cloned().collect()
    }

    /// Iterate over elements
    pub fn iter(&self) -> std::collections::hash_set::Iter<'_, T> {
        self.data.iter()
    }

    /// Union with another set
    pub fn union(&self, other: &Set<T>) -> Set<T> {
        let mut result = self.clone();
        for item in other.iter() {
            result.add(item.clone());
        }
        result
    }

    /// Intersection with another set
    pub fn intersection(&self, other: &Set<T>) -> Set<T> {
        let mut result = Set::new();
        for item in self.iter() {
            if other.contains(item) {
                result.add(item.clone());
            }
        }
        result
    }

    /// Difference with another set
    pub fn difference(&self, other: &Set<T>) -> Set<T> {
        let mut result = self.clone();
        for item in other.iter() {
            result.remove(item);
        }
        result
    }
}

impl<T: Hash + Eq + Clone> Default for Set<T> {
    fn default() -> Self {
        Self::new()
    }
}

// Trait aliases for OCCT-style naming conventions
pub type TColStd_Array1<T> = Array1<T>;
pub type TColStd_List<T> = List<T>;
pub type TColStd_Map<K, V> = Map<K, V>;
pub type TColStd_Set<T> = Set<T>;

#[cfg(test)]
mod tests {
    use super::*;

    // ===== Array1 Tests =====

    #[test]
    fn test_array1_0based_creation() {
        let mut arr: Array1<i32> = Array1::new(10);
        assert_eq!(arr.lower_bound(), 0);
        assert_eq!(arr.length(), 0);
        
        arr.append(42);
        assert_eq!(arr.length(), 1);
        assert_eq!(arr.value(0), Some(42));
    }

    #[test]
    fn test_array1_1based_creation() {
        let mut arr: Array1<i32> = Array1::new_1based(3);
        assert_eq!(arr.lower_bound(), 1);
        
        arr.append(10);
        arr.append(20);
        arr.append(30);
        
        assert_eq!(arr.length(), 3);
        assert_eq!(arr.value(1), Some(10));
        assert_eq!(arr.value(2), Some(20));
        assert_eq!(arr.value(3), Some(30));
        assert_eq!(arr.value(0), None); // Index 0 is invalid in 1-based array
    }

    #[test]
    fn test_array1_from_vec() {
        let vec = vec![1, 2, 3, 4, 5];
        let arr = Array1::from_vec(vec);
        
        assert_eq!(arr.length(), 5);
        assert_eq!(arr.value(0), Some(1));
        assert_eq!(arr.value(4), Some(5));
    }

    #[test]
    fn test_array1_set_value() {
        let mut arr: Array1<i32> = Array1::new(5);
        arr.append(1);
        arr.append(2);
        arr.append(3);
        
        assert!(arr.set_value(1, 99).is_ok());
        assert_eq!(arr.value(1), Some(99));
        
        assert!(arr.set_value(10, 0).is_err()); // Out of bounds
    }

    #[test]
    fn test_array1_insert_remove() {
        let mut arr: Array1<i32> = Array1::new(5);
        arr.append(1);
        arr.append(3);
        
        assert!(arr.insert(1, 2).is_ok());
        assert_eq!(arr.to_vec(), vec![1, 2, 3]);
        
        assert_eq!(arr.remove(1), Ok(2));
        assert_eq!(arr.to_vec(), vec![1, 3]);
    }

    #[test]
    fn test_array1_upper_bound() {
        let mut arr0: Array1<i32> = Array1::new(5);
        arr0.append(1);
        arr0.append(2);
        arr0.append(3);
        assert_eq!(arr0.upper_bound(), 2); // 0-based: length - 1
        
        let mut arr1: Array1<i32> = Array1::new_1based(3);
        arr1.append(1);
        arr1.append(2);
        arr1.append(3);
        assert_eq!(arr1.upper_bound(), 3); // 1-based: length
    }

    // ===== List Tests =====

    #[test]
    fn test_list_creation() {
        let list: List<i32> = List::new();
        assert_eq!(list.length(), 0);
        assert!(list.is_empty());
    }

    #[test]
    fn test_list_append_prepend() {
        let mut list: List<i32> = List::new();
        
        list.append(1);
        list.append(2);
        list.append(3);
        list.prepend(0);
        
        assert_eq!(list.length(), 4);
        assert_eq!(list.to_vec(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_list_value_set_value() {
        let mut list: List<String> = List::new();
        list.append("a".to_string());
        list.append("b".to_string());
        list.append("c".to_string());
        
        assert_eq!(list.value(1), Some("b".to_string()));
        
        assert!(list.set_value(1, "B".to_string()).is_ok());
        assert_eq!(list.value(1), Some("B".to_string()));
    }

    #[test]
    fn test_list_insert_remove() {
        let mut list: List<i32> = List::new();
        list.append(1);
        list.append(3);
        list.append(5);
        
        assert!(list.insert(1, 2).is_ok());
        assert_eq!(list.to_vec(), vec![1, 2, 3, 5]);
        
        assert_eq!(list.remove(1), Ok(2));
        assert_eq!(list.to_vec(), vec![1, 3, 5]);
    }

    #[test]
    fn test_list_first_last() {
        let mut list: List<i32> = List::new();
        list.append(1);
        list.append(2);
        list.append(3);
        
        assert_eq!(list.first(), Some(1));
        assert_eq!(list.last(), Some(3));
        
        assert_eq!(list.remove_first(), Some(1));
        assert_eq!(list.first(), Some(2));
        
        assert_eq!(list.remove_last(), Some(3));
        assert_eq!(list.last(), Some(2));
    }

    // ===== Map Tests =====

    #[test]
    fn test_map_creation() {
        let map: Map<String, i32> = Map::new();
        assert_eq!(map.length(), 0);
        assert!(map.is_empty());
    }

    #[test]
    fn test_map_insert_value() {
        let mut map: Map<String, i32> = Map::new();
        
        map.insert("one".to_string(), 1);
        map.insert("two".to_string(), 2);
        map.insert("three".to_string(), 3);
        
        assert_eq!(map.length(), 3);
        assert_eq!(map.value(&"two".to_string()), Some(2));
    }

    #[test]
    fn test_map_bind() {
        let mut map: Map<String, i32> = Map::new();
        
        assert_eq!(map.bind("key1".to_string(), 100), None);
        assert_eq!(map.bind("key1".to_string(), 200), Some(100));
        assert_eq!(map.value(&"key1".to_string()), Some(200));
    }

    #[test]
    fn test_map_contains() {
        let mut map: Map<i32, String> = Map::new();
        map.insert(1, "one".to_string());
        
        assert!(map.contains(&1));
        assert!(!map.contains(&2));
    }

    #[test]
    fn test_map_remove() {
        let mut map: Map<String, i32> = Map::new();
        map.insert("key".to_string(), 42);
        
        assert_eq!(map.remove(&"key".to_string()), Some(42));
        assert!(map.is_empty());
    }

    #[test]
    fn test_map_keys_values() {
        let mut map: Map<i32, String> = Map::new();
        map.insert(1, "one".to_string());
        map.insert(2, "two".to_string());
        map.insert(3, "three".to_string());
        
        let mut keys = map.keys();
        keys.sort();
        assert_eq!(keys, vec![1, 2, 3]);
        
        let mut values = map.values();
        values.sort();
        assert_eq!(values, vec!["one".to_string(), "three".to_string(), "two".to_string()]);
    }

    // ===== Set Tests =====

    #[test]
    fn test_set_creation() {
        let set: Set<i32> = Set::new();
        assert_eq!(set.length(), 0);
        assert!(set.is_empty());
    }

    #[test]
    fn test_set_add_insert() {
        let mut set: Set<i32> = Set::new();
        
        assert!(set.add(1)); // First time returns true
        assert!(set.add(2));
        assert!(set.add(3));
        
        assert!(!set.add(1)); // Second time returns false (already exists)
        assert_eq!(set.length(), 3);
    }

    #[test]
    fn test_set_contains() {
        let mut set: Set<String> = Set::new();
        set.add("apple".to_string());
        set.add("banana".to_string());
        
        assert!(set.contains(&"apple".to_string()));
        assert!(!set.contains(&"cherry".to_string()));
    }

    #[test]
    fn test_set_remove() {
        let mut set: Set<i32> = Set::new();
        set.add(1);
        set.add(2);
        set.add(3);
        
        assert!(set.remove(&2));
        assert!(!set.contains(&2));
        assert_eq!(set.length(), 2);
    }

    #[test]
    fn test_set_union() {
        let mut set1: Set<i32> = Set::new();
        set1.add(1);
        set1.add(2);
        set1.add(3);
        
        let mut set2: Set<i32> = Set::new();
        set2.add(3);
        set2.add(4);
        set2.add(5);
        
        let union = set1.union(&set2);
        assert_eq!(union.length(), 5);
        assert!(union.contains(&1));
        assert!(union.contains(&5));
    }

    #[test]
    fn test_set_intersection() {
        let mut set1: Set<i32> = Set::new();
        set1.add(1);
        set1.add(2);
        set1.add(3);
        
        let mut set2: Set<i32> = Set::new();
        set2.add(2);
        set2.add(3);
        set2.add(4);
        
        let inter = set1.intersection(&set2);
        assert_eq!(inter.length(), 2);
        assert!(inter.contains(&2));
        assert!(inter.contains(&3));
        assert!(!inter.contains(&1));
    }

    #[test]
    fn test_set_difference() {
        let mut set1: Set<i32> = Set::new();
        set1.add(1);
        set1.add(2);
        set1.add(3);
        
        let mut set2: Set<i32> = Set::new();
        set2.add(2);
        set2.add(3);
        set2.add(4);
        
        let diff = set1.difference(&set2);
        assert_eq!(diff.length(), 1);
        assert!(diff.contains(&1));
        assert!(!diff.contains(&2));
    }

    // ===== OCCT-style naming tests =====

    #[test]
    fn test_occt_type_aliases() {
        let mut arr: TColStd_Array1<i32> = TColStd_Array1::new(5);
        arr.append(42);
        assert_eq!(arr.length(), 1);
        
        let mut list: TColStd_List<String> = TColStd_List::new();
        list.append("test".to_string());
        assert_eq!(list.length(), 1);
        
        let mut map: TColStd_Map<i32, String> = TColStd_Map::new();
        map.insert(1, "one".to_string());
        assert_eq!(map.length(), 1);
        
        let mut set: TColStd_Set<i32> = TColStd_Set::new();
        set.add(42);
        assert_eq!(set.length(), 1);
    }

    #[test]
    fn test_collection_with_complex_types() {
        let mut arr: Array1<Vec<i32>> = Array1::new(3);
        arr.append(vec![1, 2, 3]);
        arr.append(vec![4, 5, 6]);
        
        assert_eq!(arr.length(), 2);
        assert_eq!(arr.value(0), Some(vec![1, 2, 3]));
        
        let mut map: Map<String, Vec<i32>> = Map::new();
        map.insert("nums".to_string(), vec![1, 2, 3]);
        assert_eq!(map.value(&"nums".to_string()), Some(vec![1, 2, 3]));
    }

    #[test]
    fn test_list_iter() {
        let mut list: List<i32> = List::new();
        list.append(1);
        list.append(2);
        list.append(3);
        
        let sum: i32 = list.iter().sum();
        assert_eq!(sum, 6);
    }

    #[test]
    fn test_array1_iter() {
        let mut arr: Array1<i32> = Array1::new(5);
        arr.append(10);
        arr.append(20);
        arr.append(30);
        
        let sum: i32 = arr.iter().sum();
        assert_eq!(sum, 60);
    }

    #[test]
    fn test_set_to_vec() {
        let mut set: Set<i32> = Set::new();
        set.add(3);
        set.add(1);
        set.add(2);
        
        let mut vec = set.to_vec();
        vec.sort();
        assert_eq!(vec, vec![1, 2, 3]);
    }

    #[test]
    fn test_list_clear() {
        let mut list: List<i32> = List::new();
        list.append(1);
        list.append(2);
        list.append(3);
        
        assert_eq!(list.length(), 3);
        list.clear();
        assert!(list.is_empty());
    }

    #[test]
    fn test_map_clear() {
        let mut map: Map<i32, String> = Map::new();
        map.insert(1, "one".to_string());
        map.insert(2, "two".to_string());
        
        assert_eq!(map.length(), 2);
        map.clear();
        assert!(map.is_empty());
    }
}
