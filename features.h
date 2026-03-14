//features.h (header file)
//Feature struct and function declarations used by all three programs.

#pragma once
#include <string>
#include <vector>
#include <set>
using namespace std;

// FeatureSet holds all features extracted from a code file. These features are used for training and evaluating the logistic regression model and for making rule based decisions.

struct FeatureSet 
{
    string filename;

    // Code size and structure features
    int total_lines;
    int non_empty_lines;
    int blank_lines_count;
    int max_line_length;
    int open_brace_count;
    int close_brace_count;
    double average_line_length;

    // Code comment behavior features
    int line_comment_count;
    int block_comment_count;
    int total_comment_lines;
    double comment_density;
    double average_comment_length;


    // Code control flow features
    int if_count;
    int else_count;
    int for_count;
    int while_count;
    int switch_count;
    int return_count;
    int control_count;


    // Code basic command features
    int include_count;
    int using_namespace_std;
    int std_prefix_count;
    int cout_count;
    int cin_count;
    int endl_count;

    // Code variable / function naming features
    double average_identifier_length;
    int unique_identifier_count;

    // Code indentation and formatting features
    double indentation_consistency;
    int magic_number_count;
    int max_nesting_depth;
    int function_count;
    int operator_diversity;

    int label; //for logistic regression model, 1 = AI-generated code, 0 = human-written code
};

// Function declaration for feature extraction.


// Extract features from a code file and return a FeatureSet struct containing all extracted features.
FeatureSet extractFeatures(const string& filepath);

// Print readable report of FeatureSet.  
void printFeatureReport(const FeatureSet& f);

// Save a FeatureSet to a CSV file. (For logistic regression dataset)
void appendtoCSV(const FeatureSet& f, const string& csvPath);


// List names of all features in the FeatureSet struct. (For logistic regression model)
vector<string> featureNames();

// Converts a FeatureSet struct into plain list of feature values (numbers) for logistic regression model to read and train on.
vector<double> featureVector(const FeatureSet& f);

// Give CSV file path, read dataset and load it into memory so logisitc regression model can train on it. X: feature numbers Y: labels 1 for AI-generated code, 0 for humman-written code. Filenames: get the name of each code filein dataset.
bool loadDataset(const string& csvPath, vector<vector<double>>& X, vector<int>& Y, vector<string>& filenames);
