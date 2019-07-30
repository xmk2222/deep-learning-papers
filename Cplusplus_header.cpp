#include<iostream> //getline(cin, )
#include<algorithm> // sort(a.begin(), a.end(), greater<int>()) reverse(a.begin(), a.end())
#include<cmath> // pow(double x, double y)
#include<limits.h> // INT_MAX, INT_MIN
#include<vector> //push_back(), myvec.erase(pos_iter), myvec.erase(start_iter, end_iter)
// pair<class T, class T>, mypair.first, mypair.second
#include<list> //push_back(), push_front(), mylist.sort(), mylist.insert(pos_iter, ele_num, ele), mylist.erase(iter), mylist.erase(iter_first, iter_last)
#include<deque> //push_back(), push_front()
#include<queue> //push(), pop()
// priority_queue<>, first element is the greatest, top(), push(), pop()
// priority_queue<int, vector<int>, greater<int> > 小顶堆 priority_queue<int, vector<int>, less<int> > 大顶堆（默认）
#include<stack> //top(), push(), pop()
#include<string> //substr(pos, len), count(str.begin(), str.end(), ch)
#include<set> //insert(), erase(), count(), find()
#include<map> //count(), erase(key), insert({key, element})
#include<unordered_set>
#include<unordered_map>
#include<iterator>
using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};
 
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};
 
static bool compare(int n1, int n2) {return n1 > n2;}

#include <sstream>
vector<string> split(string s, char delimiter) {
    vector<std::string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter))
    {
        tokens.push_back(token);
    }
   return tokens;
}

int main() {
    string str;
    getline(cin, str);
    stringstream ss(str);
    int input;
    vector<int> inputs;
    //char delimiter;
    while(ss >> input) {
        inputs.push_back(input);
        //ss >> delimiter;
    }
    return 0;
}
