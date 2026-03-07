#include "features.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <regex>
#include <set>
#include <cmath>
#include <map>
#include <sys/stat.h>
using namespace std;

// Internal help functions for feature extraction


// haystack is the source code that we are searching, needle is the substring we are looking for.
// this function count the number of occurances appears in the source code and return the count as an integer. 
static int countSubstr(const string& haystack, const string& needle)
{
    if (needle.empty()) return 0;
    int count = 0;
    size_t pos = 0;
    while ((pos = haystack.find(needle, pos)) != string::npos)
    {
        ++count;
        pos += needle.size();
    }

    return count;
}


// function counts the number of occurrences of a keyword in the source code. Ensures that the keyword is matched as a whole world and not as part of another word. 
// src = source code, kw = keyword we want to count.
// ex. searches for "for" will not count words like "format" or "before". 
static int countKeyword(const string& src, const string& kw)
{
    try
    {
        regex re("\\b" + kw + "\\b");
        auto begin = sregex_iterator(src.begin(), src.end(), re);
        auto end = sregex_iterator();
        return (int) distance(begin, end);
    }
    catch (...) {return 0;}
}

// function removes block comments (/* ... */) from src leaving only the pure code.
// result = the source code with block comments removed.

static string removeBlockComments(const string& src)
{
    string result;
    result.reserve(src.size());
    size_t i = 0;
    while (i < src.size())
    {
        if (i + 1 < src.size() && src[i] == '/' && src[i + 1] == '*')
        {
            // skip until end of block comment
            i += 2;
            while (i + 1 < src.size() && !(src[i] == '*' && src[i + 1] == '/'))
            ++i;
            i += 2; // skip the closing */
        }
        else
        {
            result += src[i++];
        }
    }

    return result;
}

// function behaves similar to removeBlockComments but removes line comments instead (//... to end of comment).
// result = source code with line comments removed. 
static string removeLineComments(const string& src)
{
    string result;
    result.reserve(src.size());
    size_t i = 0;
    while (i < src.size())
    {
        if (i + 1 < src.size() && src[i] == '/' && src[i + 1] == '/')
        {
            while (i < src.size() && src[i] != '\n') ++i;
        }
        else
        {
            result += src[i++];
        }
    }

    return result;
}


// words to ignore when counting unique identifiers.
static const set<string> CPP_KEYWORDS = {
    "int", "float", "double","char","bool","void","string","long","short",
    "unsigned","signed","const","static","return","if","else","for",
    "while","do","switch","case","break","continue","class","struct",
    "public","private","protected","new","delete","true","false",
    "nullptr","include","using","namespace","std","main","endl",
    "cout","cin","auto","this","template","typename","vector","push_back",
    "size","begin","end","sort","pair","map","set","queue","stack"
};


// --feature extraction functions:--
// opens and read the source .cpp file, extracts all features defined in the FeatureSet struct and return a FeatureSet object with all features included.


// --read file into memory, extrat file name, split into lines.--
FeatureSet extractFeatures(const string& filepath)
{
    FeatureSet f;
    f.label = -1; // unknown by default. for labaled dataset.

    // extract filename from path
    size_t slash = filepath.find_last_of("/\\");
    f.filename = (slash == string::npos) ? filepath : filepath.substr(slash + 1);

    // read entire file
    ifstream file(filepath);
    if (!file.is_open())
    {
        cerr << "[!] Failed to open file: " << filepath << endl;
        return f;
    }

    string source((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    file.close();

    // split source into lines.
    vector<string> lines;
    {
        istringstream ss(source);
        string line;
        while (getline(ss, line))
        {
            lines.push_back(line);
        }
    }

    // --size and structure features.--
    // sumLen = total length of all lines combined.
    // max_line_length = length of the longest line in source code, will be updated after iterating through all lines.
    // indentedLines = number of lines with proper indentation. (start with space or tab)
    f.total_lines = (int) lines.size();
    f.blank_lines_count = 0;

    long long sumLen = 0;
    f.max_line_length = 0;
    int indentedLines = 0;

    for (const auto& line : lines)
    {
        string trimmed = line;
        size_t start = trimmed.find_first_not_of(" \t\r");
        // if start is npos it means line is empty or contains only whitespace, this gets counted as blank line and we continue to next line.
        if (start == string::npos)
        {
            f.blank_lines_count++;
            continue;
        }

        // check indentation consistency.
        int len = (int) line.size();
        sumLen += len;
        f.max_line_length = max(f.max_line_length, len);
    }

    f.non_empty_lines = f.total_lines - f.blank_lines_count;
    f.average_line_length = f.non_empty_lines > 0 ? (double) sumLen / f.non_empty_lines : 0.0;
    // indentation consistency = number of indented lines / total non empty lines. 
    // 1 = lines are properly indented, 0 = no indentation at all. (could be sign of AI-generated code)
    f.indentiation_consistency = f.non_empty_lines > 0 ? (double) indentedLines / f.non_empty_lines : 0.0;

    f.open_brace_count = countSubstr(source, "{");
    f.close_brace_count = countSubstr(source, "}");

    // --comment behavior features.--
    // identifies comment behavior and stores in the FeatureSet struct.
    f.line_comment_count = countSubstr(source, "//");
    f.block_comment_count = countSubstr(source, "/*");

    f.total_comment_lines = 0;
    for (const auto& line : lines)
    {
        string t = line;
        size_t s = t.find_first_not_of("\t");
        if (s == string::npos) continue;
        string trimmed = t.substr(s);
        if (trimmed.substr(0, 2) == "//" || trimmed.substr(0, 2) == "/*" || (!trimmed.empty() && trimmed[0] == '*'))
        f.total_comment_lines++;
    }

    // calculates comment density = total comment lines. total lines in source code. 
    // higher comment density = likely to be AI-generated code. 
    f.comment_density = f.total_lines > 0 ? (double) f.total_comment_lines / f.total_lines : 0.0;

    // average comment length
    long long commentCharSum = 0;
    int commentCount = 0;
    {
        regex lineCommentRe("//(.+)");
        auto it = sregex_iterator(source.begin(), source.end(), lineCommentRe);
        auto end = sregex_iterator();
        for (; it != end; ++it)
        {
            string text = (*it)[1].str();
            commentCharSum += (int) text.size();
            commentCount++;
        }
    }

    // calculates average comment length.
    // higher average = likely to be AI-generated code.
    f.average_comment_length = commentCount > 0 ? (double) commentCharSum / commentCount : 0.0;

    // strip comments of furhter analysis 
    string codeOnly = removeLineComments(removeBlockComments(source));

    // --control flow features.--
    

}
