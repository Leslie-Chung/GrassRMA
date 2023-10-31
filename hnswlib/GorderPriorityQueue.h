#include <cstdlib>
#include <vector>
#include <climits>
#include <algorithm>
#include <unordered_map>
#include <iostream>
#include <utility> 



template<typename node_id_t>
class GorderPriorityQueue{

	typedef std::unordered_map<node_id_t, int> map_t;

	struct Node {
		node_id_t key;
		int priority;
	};

	std::vector<Node> list;
	map_t index_table; // map: key -> index in list

	inline void swap(int i, int j){
		Node tmp = list[i];
		list[i] = list[j];
		list[j] = tmp;
		index_table[list[i].key] = i;
		index_table[list[j].key] = j;
	}


	public:
	GorderPriorityQueue(const std::vector<node_id_t>& nodes){
		for (int i = 0; i < nodes.size(); i++){
			list.push_back({nodes[i],0});
			index_table[nodes[i]] = i;
		}
	}

	GorderPriorityQueue(size_t N){
		for (int i = 0; i < N; i++){
			list.push_back({i,0});
			index_table[i] = i;
		}
	}


	void print(){
		for(int i = 0; i < list.size(); i++){
			std::cout<<"("<<list[i].key<<":"<<list[i].priority<<")"<<" ";
		}
		std::cout<<std::endl;
	}


	static bool compare(const Node &a, const Node &b){
		return (a.priority < b.priority);
	}


	void increment(node_id_t key){
		typename map_t::const_iterator i = index_table.find(key);
		if (i == index_table.end()){
			return;
		}

		auto it = std::upper_bound(list.begin(), list.end(), list[i->second], compare);
		size_t new_index = it - list.begin() - 1; // possible bug
		swap(i->second, new_index);
		list[new_index].priority++;
	}

	void decrement(node_id_t key){
		typename map_t::const_iterator i = index_table.find(key);
		if (i == index_table.end()){
			return;
		}

		auto it = std::lower_bound(list.begin(), list.end(), list[i->second], compare);
		size_t new_index = it - list.begin(); // POSSIBLE BUG


		swap(i->second, new_index);
		list[new_index].priority--;
	}
	
	node_id_t pop(){
		Node max = list.back();
		list.pop_back();
		index_table.erase(max.key);
		return max.key;
	}

	size_t size(){
		return list.size();
	}

};
