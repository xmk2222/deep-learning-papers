Mostly used C++ functions

# Functions

## subvector

```C++
vector<T>::const_iterator first = myVec.begin() + 100000;
vector<T>::const_iterator last = myVec.begin() + 101000;
vector<T> newVec(first, last);
```

## substr

```C++
string substr (size_t pos = 0, size_t len = npos) const;
```

## heap

make_heap

pop_heap

push_heap

sort_heap

```C++
int myints[] = {10,20,30,5,15};
std::vector<int> v(myints,myints+5);
std::make_heap (v.begin(),v.end());
std::pop_heap (v.begin(),v.end()); 
v.pop_back();
v.push_back(99); 
std::push_heap (v.begin(),v.end());
std::sort_heap (v.begin(),v.end());
```

# Classes

## string

### constructor
1|2
-----|-----
default (1)	          |string();
copy (2)	            |string (const string& str);
substring (3)	        |string (const string& str, size_t pos, size_t len = npos);
from c-string (4)	    |string (const char* s);
from buffer (5)	      |string (const char* s, size_t n);
fill (6)	            |string (size_t n, char c);
range (7)	            |template <class InputIterator>string  (InputIterator first, InputIterator last);
initializer list (8)	|string (initializer_list<char> il);
move (9)	            |string (string&& str) noexcept;
  
### Modifiers
1|2|3
-----|-----|-----
operator+=  |Append to string (public member function )             |
append      |Append to string (public member function )             |
push_back   |Append character to string (public member function )   |void push_back (char c);
insert      |Insert into string (public member function )           |string& insert (size_t pos, const string& str);
erase       |Erase characters from string (public member function ) |[\*](*erase)
replace     |Replace portion of string (public member function )    |
swap        |Swap string values (public member function )           |void swap (string& str);
pop_back    |Delete last character (public member function )        |

### operations
1|2
-----|-----
find               |Find content in string (public member function )
rfind              |Find last occurrence of content in string (public member function )
find_first_of      |Find character in string (public member function )
find_last_of       |Find character in string from the end (public member function )
find_first_not_of  |Find absence of character in string (public member function )
find_last_not_of   |Find non-matching character in string from the end (public member function )
substr             |Generate substring (public member function )
compare            |Compare strings (public member function )

#### erase
1|2
-----|-----
sequence (1)  |string& erase (size_t pos = 0, size_t len = npos);
character (2) |iterator erase (const_iterator p);
range (3)     |iterator erase (const_iterator first, const_iterator last);
