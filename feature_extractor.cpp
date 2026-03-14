// feature extractor = reads c++ source code files, extracts 32 measurable features, and prints a formatted report and appends results to a CSV file. 

#include "features.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
using namespace std;
namespace fs = filesystem;

// parse command line arguments
// Args struct = holds command line arguments for feature extraction. 
// target = file or directory path to extract features from.
// label = -1 = user doesn't specify, 0 = human-written code, 1 = AI-generated code. (for logistic regression dataset labeling)
// csvPath = name of CSV file to save features into. defaults to "features.csv" if user doesn't specify.
// quiet = if true program skips printing feature report to console, and only saves to CSV file. 

struct Args
{
    string target; //file or directory path
    int label = -1;
    string csvPath = "features.csv";
    bool quiet = false;
};

// reads user input from command line and returns an Args struct with all arguments parsed and stored. 
static Args parseArgs(int argc, char* argv[])
{
    Args a;
    if (argc < 2) return a;
    a.target = argv[1];

    for (int i = 2; i < argc; i++)
    {
        string arg = argv[i];
        if (arg == "0" || arg == "1")
        {
            a.label = stoi(arg);
        }
        else if (arg.substr(0, 6) == "--csv=")
        {
            a.csvPath = arg.substr(6);
        }
        else if(arg == "--quiet" || arg == "-q")
        {
            a.quiet = true;
        }
    }
    
    return a;
}

// process a single file
// extracts features from file, prints report to console (if not quiet), and appends features to CSV file. 

static void processFile(const string& path, int label, const string& csvPath, bool quiet)
{
    FeatureSet f= extractFeatures(path);
    f.label = label;

    if (!quiet) printFeatureReport(f);

    appendtoCSV(f, csvPath);

    cout << "[:)] Processed: " << f.filename << " --> " << csvPath << (label >= 0 ? ("  (label=" + to_string(label) + ")") : "") << endl;
}

// main function 
// argc = number of command line arguments

int main(int argc, char* argv[])
{
    if (argc< 2)
    {
        cout << "Usage: " << endl; 
        cout << "   ./feature_extractor <file.cpp> [label] [--csv=out.csv]" << endl;
        cout << "   ./feature_extractor <folder/> [label] [--csv=out.csv]" << endl;
        cout << "   label: 1 = AI-generated, 0  = human-written" << endl;
        cout << "   (leave blank if just inspecting file)" << endl;
        return 1;
    }

    Args args = parseArgs(argc, argv);

    // determine if target is a file or directory
    if (fs::is_directory(args.target))
    {
        int processed = 0;
        for(const auto& entry : fs::directory_iterator(args.target))
        {
            if (entry.path().extension() == ".cpp")
            {
                processFile(entry.path().string(), args.label, args.csvPath, args.quiet);

                processed++;
            }
        }

        if (processed == 0)
        {
            cout << "[!] no .cpp files found in: " << args.target << endl;
        }
        else
        {
            cout << "[:)] Processed " << processed << " files --> " << args.csvPath << endl;    
        }
    }

    else if (fs::is_regular_file(args.target))
    {
        if (args.target.size() < 4 || args.target.substr(args.target.size() -4) != ".cpp")
        {
            cerr << "[!] File must be a .cpp source code file." << endl;
            return 1;
        }

        processFile(args.target, args.label, args.csvPath, false);

    }

    else
    {
        cerr << "[!] Path not found: " << args.target << endl;
        return 1;
    }

    return 0;
}
