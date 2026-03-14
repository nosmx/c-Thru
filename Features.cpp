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
    FeatureSet f = {};
    f.label = -1; // unknown by default. for labaled dataset.

    // extract filename from path
    size_t slash = filepath.find_last_of("/\\");
    f.filename = (slash == string::npos) ? filepath : filepath.substr(slash + 1);

    // read entire file
    ifstream file(filepath);
    cout << "Trying to open: " << filepath << endl;
    if (!file.is_open())
    {
        cerr << "[!] Failed to open file: " << filepath << endl;
        return f;
    }

    string source;
    string fileline;
    while (getline(file, fileline))
    {
        source += fileline + "\n";
    }
    file.close();

    // TEMPORARY DEBUG LINE
    cout << "Source length: " << source.size() << " characters" << endl;

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
        if (line.size() >= 4 && (line.substr(0, 4) == "    " || line[0] == '\t'))
        {
            indentedLines++;
        }

        int len = (int) line.size();
        sumLen += len;
        f.max_line_length = max(f.max_line_length, len);
    }

    f.non_empty_lines = f.total_lines - f.blank_lines_count;
    f.average_line_length = f.non_empty_lines > 0 ? (double) sumLen / f.non_empty_lines : 0.0;
    // indentation consistency = number of indented lines / total non empty lines. 
    // 1 = lines are properly indented, 0 = no indentation at all. (could be sign of AI-generated code)
    f.indentation_consistency = f.non_empty_lines > 0 ? (double) indentedLines / f.non_empty_lines : 0.0;

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
        size_t s = t.find_first_not_of(" \t");
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
    // counts control flow keywords only from source code with comments removed. (codeOnly)
    f.if_count = countKeyword(codeOnly, "if");
    f.else_count = countKeyword(codeOnly, "else");
    f.for_count = countKeyword(codeOnly, "for");
    f.while_count =  countKeyword(codeOnly, "while");
    f.switch_count =  countKeyword(codeOnly, "switch");
    f.return_count = countKeyword(codeOnly, "return");
    f.control_count = f.if_count + f.else_count + f.for_count + f.while_count + f.switch_count;

    // --basic command features.--
    f.include_count = countSubstr(source, "#include");
    f.using_namespace_std = (source.find("using namespace std") != string::npos) ? 1 : 0;
    f.std_prefix_count = countSubstr(source, "std::");
    f.cout_count = countKeyword(codeOnly, "cout");
    f.cin_count = countKeyword(codeOnly, "cin");
    f.endl_count = countKeyword(codeOnly, "endl");

    // --variable and function naming features.--
    {

    regex idRe("\\b([a-zA-Z_][a-zA-Z0-9_]*)\\b"); // regex pattern to match C++ variables and function names.
    set<string> uniqueIds; // collect unique identifiers
    auto it = sregex_iterator(codeOnly.begin(), codeOnly.end(), idRe);
    auto end = sregex_iterator();
    for (; it != end; ++it) // sorts through all identifiers found in codeOnly and adds unique ones to the uniqueIds set while ignoring c++ keywords.
    {
        string id = (*it)[1].str();
        if (CPP_KEYWORDS.find(id) == CPP_KEYWORDS.end())
        {
            uniqueIds.insert(id);
        }
    }

    f.unique_identifier_count = (int)uniqueIds.size();
    long long idLenSum = 0;
    for (const auto& id : uniqueIds) idLenSum += (int)id.size();
    f.average_identifier_length = f.unique_identifier_count > 0 ? (double)idLenSum / f.unique_identifier_count : 0.0;

    }

    // --indentation and formatting features.--
    // magic numbers: numeric literals in code that are not a part of identifiers or floats.
   
    {
        int count = 0;
        for (size_t i = 0; i < codeOnly.size(); i++)
        {
            if (isdigit((unsigned char)codeOnly[i]))
            {
                // check if previous char is not alphanumeric, underscore or dot.
                bool prevOk = (i == 0) || (!isalnum((unsigned char)codeOnly[i-1]) && codeOnly[i-1] != '_' && codeOnly[i-1] != '.');
                // skip past the whole number
                size_t j = i;
                while (j < codeOnly.size() && isdigit((unsigned char)codeOnly[j])) j++;
                // check if next char is not alphanumeric, underscore or dot.
                bool nextOk = (j >= codeOnly.size()) || (!isalnum((unsigned char)codeOnly[j]) && codeOnly[j] != '_' && codeOnly[j] != '.');
                if (prevOk && nextOk) count++;
                i = j -1;
            }
        }

        f.magic_number_count = count; 
    }

    // Max nesting depth.
    // depth = variable to count nesting depth

    f.max_nesting_depth = 0;
    {
        int depth = 0;

        for (char ch : source)
        {
            if (ch == '{') 
            {
                depth++; f.max_nesting_depth = max(f.max_nesting_depth, depth);
            }
            else if (ch == '}') 
            {
                depth = max(0, depth -1);       
            }
        }
        
    }

    // function count
    // count pattern that matches C++ function definitions

    {
        regex fnRe("\\b(int|void|double|float|bool|char|string|long)" "\\s+\\w+\\s*\\(");
        auto it = sregex_iterator(codeOnly.begin(), codeOnly.end(), fnRe);
        auto end = sregex_iterator();
        f.function_count = (int) distance(it, end);
    }

    // operator diversity
    // ops = set to collect unique operators.

    {
        set<string> ops;
        regex opRe("[+\\-*/%=<>!&|^~]+");
        auto it = sregex_iterator(codeOnly.begin(), codeOnly.end(), opRe);
        auto end = sregex_iterator();

        for (; it != end; ++it)
        {
            ops.insert((*it)[0].str()); 
        }
        f.operator_diversity = (int) ops.size();
    }

    return f;
}



// output features report in readable format.
// display each feature in readable format.
// irow = function to print integer values
// drow = function to print double values with precision

void printFeatureReport(const FeatureSet& f)
{
    auto sep = []() {cout << string (58, '=')  << endl;};

    // integer row

    auto irow = [](const string& name, int val)
    {
        cout << "   " << left << setw(32) << name << val << endl;
    };

    //double row (with configurable precision)

    auto drow = [](const string& name, double val, int prec)
    {
        ostringstream oss;
        oss << fixed << setprecision(prec) << val;
        cout << "   " << left << setw(32) << name << oss.str() << endl;
    };

    sep();
    cout << "   Feature Report: " << f.filename << endl;
    sep();

    cout << "\n  [Size / Structure Features]\n";
    irow("total lines", f.total_lines);
    irow("non_empty_lines", f.non_empty_lines);
    irow("blank_lines_count", f.blank_lines_count);
    drow("average_line_length", f.average_line_length, 2);
    irow("max_line_length", f.max_line_length);
    irow("open_brace_count", f.open_brace_count);
    irow("close_brace_count", f.close_brace_count);

    cout << "\n   [Comment Behavior Features]\n";
    irow("line_comment_count", f.line_comment_count);
    irow("block_comment_count", f.block_comment_count);
    irow("total_comment_lines", f.total_comment_lines);
    drow("comment_density", f.comment_density, 4);
    drow("average_comment_length", f.average_comment_length, 2);

    cout << "\n   [Control Flow Features]\n";
    irow("if_count", f.if_count);
    irow("else_count", f.else_count);
    irow("for_count", f.for_count);
    irow("while_count", f.while_count);
    irow("switch_count", f.switch_count);
    irow("return_count", f.return_count);
    irow("control_total", f.control_count);

    cout << "\n   [Basic Command Features]\n";
    irow("include_count", f.include_count);
    irow("using_namespace_std", f.using_namespace_std);
    irow("std_prefix_count", f.std_prefix_count);
    irow("cout_count", f.cout_count);
    irow("cin_count", f.cin_count);
    irow("endl_count", f.endl_count);

    cout << "\n   [Variable / Function Naming Features]\n";
    drow("average_identifier_length", f.average_identifier_length, 2);
    irow("unique_identifier_count", f.unique_identifier_count);

    cout << "\n   [Style / Complexity Features]\n";
    drow("indentation_consistency", f.indentation_consistency, 4);
    irow("magic_number_count", f.magic_number_count);
    irow("max_nesting_depth", f.max_nesting_depth);
    irow("function_count", f.function_count);
    irow("operator_diversity", f.operator_diversity);

    sep();
    cout << endl;

}


// CSV tools for exporting features to CSV file
// CSV_HEADER = vector of column names for CSV file
// appendToCSV = function that takes a FeatureSet and file path, appends the features into CSV format.
static const vector<string> CSV_HEADER = {
    "filename",
    "total_lines",
    "non_empty_lines",
    "blank_lines_count",
    "average_line_length",
    "max_line_length",
    "open_brace_count",
    "close_brace_count",
    "line_comment_count",
    "block_comment_count",
    "total_comment_lines",
    "comment_density",
    "average_comment_length",
    "if_count",
    "else_count",
    "for_count",
    "while_count",
    "switch_count",
    "return_count",
    "control_total",
    "include_count",
    "using_namespace_std",
    "std_prefix_count",
    "cout_count",
    "cin_count",
    "endl_count",
    "average_identifier_length",
    "unique_identifier_count",
    "indentation_consistency",
    "magic_number_count",
    "max_nesting_depth",
    "function_count",
    "operator_diversity",
    "label"
};

void appendtoCSV(const FeatureSet& f, const string& csvPath)
{
    // check if file exist (to decide whether to write header or not)
    bool fileExists = false;
    {
        ifstream test(csvPath);
        fileExists = test.good();
    }

    ofstream out(csvPath, ios::app);
    if (!out.is_open())
    {
        cerr << "[!] Cannot open CSV: " << csvPath << endl;
        return;
    }

    if (!fileExists)
    // write header row
    {
        for (size_t i = 0; i < CSV_HEADER.size(); i++)
        {
            out << CSV_HEADER[i];
            if (i + 1 < CSV_HEADER.size()) out << ",";
        }

        out << endl;

    }


// write data row
// writes all features of the FeatureSet struct into the CSV file in the same order as defined in the CSV_HEADER vector.

out << f.filename                   << ","
    << f.total_lines                << ","
    << f.non_empty_lines            << ","
    << f.blank_lines_count          << ","
    << fixed << setprecision(4) << f.average_line_length        << ","
    << f.max_line_length            << ","
    << f.open_brace_count           << ","
    << f.close_brace_count          << ","
    << f.line_comment_count         << ","
    << f.block_comment_count        << ","
    << f.total_comment_lines        << ","
    << fixed << setprecision(6) << f.comment_density            << ","
    << fixed << setprecision(4) << f.average_comment_length     << ","
    << f.if_count                   << ","
    << f.else_count                 << ","
    << f.for_count                  << ","
    << f.while_count                << ","
    << f.switch_count               << ","
    << f.return_count               << ","
    << f.control_count              << ","
    << f.include_count              << ","
    << f.using_namespace_std        << ","
    << f.std_prefix_count           << ","
    << f.cout_count                 << ","
    << f.cin_count                  << ","
    << f.endl_count                 << ","
    << fixed << setprecision(4) << f.average_identifier_length  << ","
    << f.unique_identifier_count    << ","
    << fixed << setprecision(6) << f.indentation_consistency    << ","
    << f.magic_number_count         << ","
    << f.max_nesting_depth          << ","
    << f.function_count             << ","
    << f.operator_diversity         << ","
    << f.label                      << endl; // label is for supervised learning, 1 = AI-generated code, 0 = human-written code.

}

// featureNames / featureVector 
// featureNames = returns a vector of feature names in the same order as defined in the CSV_HEADER vector. This is useful for logistic regression model to show which features are most important in determining whether code is AI-generated or human written.
// featureVector = takes FeatureSet and converts it into numbers since logistic regression model relies on numerical input. Function converts each feature into a double value for percision and consistency in inputs.

vector<string> featureNames()
{
    return
    {
        "total_lines",
        "non_empty_lines",
        "blank_lines_count",
        "average_line_length",
        "max_line_length",
        "open_brace_count",
        "close_brace_count",
        "line_comment_count",
        "block_comment_count",
        "total_comment_lines",
        "comment_density",
        "average_comment_length",
        "if_count",
        "else_count",
        "for_count",
        "while_count",
        "switch_count",
        "return_count",
        "control_total",
        "include_count",
        "using_namespace_std",
        "std_prefix_count",
        "cout_count",
        "cin_count",
        "endl_count",
        "average_identifier_length",
        "unique_identifier_count",
        "indentation_consistency",
        "magic_number_count",
        "max_nesting_depth",
        "function_count",
        "operator_diversity"
    };
}

vector<double> featureVector(const FeatureSet& f)
{
    return
    {
        (double) f.total_lines,
        (double) f.non_empty_lines,
        (double) f.blank_lines_count,
        f.average_line_length,
        (double) f.max_line_length,
        (double) f.open_brace_count,
        (double) f.close_brace_count,
        (double) f.line_comment_count,
        (double) f.block_comment_count,
        (double) f.total_comment_lines,
        f.comment_density,
        f.average_comment_length,
        (double) f.if_count,
        (double) f.else_count,
        (double) f.for_count,
        (double) f.while_count,
        (double) f.switch_count,
        (double) f.return_count,
        (double) f.control_count,
        (double) f.include_count,
        (double) f.using_namespace_std,
        (double) f.std_prefix_count,
        (double) f.cout_count,
        (double) f.cin_count,
        (double) f.endl_count,
        f.average_identifier_length,
        (double) f.unique_identifier_count,
        f.indentation_consistency,
        (double) f.magic_number_count,
        (double) f.max_nesting_depth,
        (double) f.function_count,
        (double) f.operator_diversity
    };
}

// loadDataset = extract information from CSV file and organize into feature numbers (x) and labels (y), and returns a pair of vectors for logistic regression model training. 
// x = feature numbers, y = labels (1 for AI-generated code, 0 for human-written code).
// cols = vector of column names in CSV file, used to map features to correct columns in CSV.
// featIdx = vector of indices for each feature in the CSV file, used to extract correct values for each feature from CSV rows. 
// fields = vector of values for each column in a CSV row, used to extract feature values and lebel for each sample. 

bool loadDataset(const string& csvPath, vector<vector<double>>& X, vector<int>& Y, vector<string>& filenames)
{
    ifstream in(csvPath);
    if (!in.is_open())
    {
        cerr << "[!] Cannot open dataset: " << csvPath << endl;
        return false;
    }

    string headerLine;
    if (!getline(in, headerLine)) { return false; }

    // parse header to find column indices
    vector<string> cols;
    {
        istringstream ss(headerLine);
        string col;
        while (getline(ss, col, ',')) cols.push_back(col);
    }

    // find index of each expected feature column
    auto names = featureNames();
    vector<int> featIdx(names.size(), -1);
    int labelIdx = -1, fileIdx = -1;

    for (int c = 0; c < (int)cols.size(); c++)
    {
        if (cols[c] == "label") {labelIdx = c; continue;}
        if (cols[c] == "filename") {fileIdx = c; continue;}

        for (int fi = 0; fi < (int)names.size(); fi++)
        {
            if (cols[c] == names[fi])
            {
                featIdx[fi] = c;
                break;
            }
        }
    }

    if (labelIdx < 0)
    {
        cerr << "[!] CSV missing 'label' column." << endl;
        return false;
    }

    string line;
    int loaded = 0, skipped = 0;
    while (getline(in, line))
    {
        if (line.empty()) continue;
        vector<string> fields;
        {
            istringstream ss(line);
            string field;
            while (getline(ss, field, ',')) fields.push_back(field);
        }

        if ((int)fields.size() <= labelIdx) {skipped++; continue;}

        int lbl = stoi(fields[labelIdx]);
        if (lbl < 0) {skipped++; continue;} // unlabeled data is skipped

        vector<double> row(names.size(), 0.0);
        for (int fi = 0; fi < (int)names.size(); fi++)
        {
            int ci = featIdx[fi];
            if (ci >= 0 && ci < (int)fields.size() && !fields[ci].empty())
            {
                row[fi] = stod(fields[ci]);
            }
        }

        X.push_back(row);
        Y.push_back(lbl);

        if (fileIdx >= 0 && fileIdx < (int)fields.size())
        {
            filenames.push_back(fields[fileIdx]);
        }
        else
        {
            filenames.push_back("sample_" + to_string(loaded));
        }
        loaded++;
    }

    cout << "[:)] Loaded " << loaded << " samples (" << skipped << " skipped)" << endl;

    return loaded > 0;
}
