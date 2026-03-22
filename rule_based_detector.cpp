// rule-based dectector, scores c++ file against 16 structural from known AI code patterns and outputs an AI probability score. 

// compile: g++ -std=c++17 -O2 rule_based_detector.cpp features.cpp -o rule_based_detector
// usage: ./rule_based_detector <file.cpp>
//        ./rule_based_detector <file.cpp> --verbose (to print detailed results)

#include "features.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <functional>
#include <algorithm>
#include <numeric>
using namespace std;

// rule definition

struct Rule
{
    string name;
    string description; 
    int weight;
    function<bool(const FeatureSet&)> test;
};

// ruleset
// each rule returns true when the feature value matches a pattern
// typical of AI-generated code
// all weights sum up to 100

static const vector<Rule> RULES = {


    // comment behavior
    {
        "High Comment Density",
        "AI over-comments: more than 25% of lines are comments.",
        12,
        [](const FeatureSet& f) { return f.comment_density > 0.25; }
    },
    {
        "Long Average Comment",
        "AI writes verbose, sentence-style comments (avg > 40 chars).",
        8,
        [](const FeatureSet& f) { return f.average_comment_length > 40.0; }
    },
    {
        "Block Comments Present",
        "AI often adds /* */ header blocks, human rarely do that for simple code.",
        5,
        [](const FeatureSet& f) { return f.block_comment_count >= 1; }
    },

    //identifier styles
    {
        "Long Variable Names",
        "AI uses descriptive names (ex. totalSumOfElements vs sum).",
        10,
        [](const FeatureSet& f) { return f.average_identifier_length > 8.0; }
    },
    {
        "High Identifier Diversity",
        "AI create many uniquely named variables, humans reuse names.",
        6,
        [](const FeatureSet& f) {return f.unique_identifier_count > 20; }
    },

    // structural cleanliness
    {
        "Very High Identation Consistency",
        "AI code is perfectly indented, human code is not as consistent.",
        10,
        [](const FeatureSet& f) { return f.indentation_consistency > 0.90; }
    },
    {
        "Low Magic Number Count",
        "AI avoids bare numeric literals (2 or fewer in the whole file).",
        5,
        [](const FeatureSet& f) { return f.magic_number_count <= 2; }
    },
    {
        "Perfectly Balanced Braces",
        "AI always produces matched brace pairs, human typos can lead to mismatches.",
        4,
        [](const FeatureSet& f) { return f.open_brace_count == f.close_brace_count; }
    },

    // control flow patterns
    {
        "Symmetric If-Else Pairs",
        "AI pairs every if with an else, humans often skip the else for simple conditions.",
        7,
        [](const FeatureSet& f)
        {
            if (f.if_count == 0) return false;
            return (double)f.else_count / f.if_count > 0.75;
        }
    },
    {
        "Moderate Control Density",
        "AI uses a predictable moderate number of control structures (3-15).",
        5,
        [](const FeatureSet& f) { return f.control_count >= 3 && f.control_count <= 15; }
    },

    // I/O and namespace patterns
    {
        "Uses 'using namespace std'",
        "AI almost always adds 'using namespace std' in beginner c++ code.",
        6,
        [](const FeatureSet& f) {return f.using_namespace_std == 1; }
    },
    {
        "Prefers endl over '\\n'",
        "AI strongly prefers endl; human programmers often mix usage of endl and '\\n'.",
        5,
        [](const FeatureSet& f) { return f.endl_count >= f.cout_count; }
    },

    // size signals
    {
        "Moderate Program Size",
        "AI CS1 programs cluster between 25-90 non-empty lines.",
        5,
        [](const FeatureSet& f) { return f.non_empty_lines >= 25 && f.non_empty_lines <= 90; }
    },
    {
        "Low Blank Line Ratio",
        "AI inserts fewer blank lines (15%) than the typical human,",
        4,
        [](const FeatureSet& f )
        {
            if (f.total_lines == 0) return false;
            return (double)f.blank_lines_count /f.total_lines < 0.15;
        }
    },

    // complexity
    {
        "Shallow Nesting Depth",
        "AI avoids deep nesting, CS1 code stays at 3 or less levels of nesting.",
        4,
        [](const FeatureSet& f) { return f.max_nesting_depth <= 3; }
    },
    {
        "Low Operator Diversity",
        "AI uses fewer distinct operator symbols than human programmers.",
        4,
        [](const FeatureSet& f) { return f.operator_diversity < 8; }
    },

    
};

// verify weight sum at compile-time via static_assert would need constexpr;
// for now we just check at runtime.

static int totalWeight()
{
    int w = 0;
    for (const auto& r : RULES) w += r.weight;
    return w;
}

// scoring engine

struct RuleResult
{
    string name;
    string description;
    int weight;
    bool fired;
};

static vector<RuleResult> evaluateRules(const FeatureSet& f)
{
    vector<RuleResult> results;
    for (const auto& rule : RULES)
    {
        results.push_back({rule.name, rule.description, rule.weight, rule.test(f)});
    }

    return results;

}

static double computeScore(const vector<RuleResult>& results)
{
    int fired = 0;
    for (const auto& r : results) if (r.fired) fired += r.weight;
    return (double)fired / totalWeight();
}

static string verdict(double score)
{
    if (score >= 0.70) return "Likely AI-generated";
    if (score >= 0.45) return "Uncertain";
    return "Likely Human-written";
}

// output

static void printReport(const FeatureSet& f, const vector<RuleResult>& results, double score, bool verbose)
{
    string LINE(67, '=');

    if (verbose)
    {
        cout << "\n" << LINE << "\n";
        cout << " Rule-Based AI Detection Report: " << f.filename << "\n";
        cout << LINE << "\n\n";

        cout << "  " << left << setw(40) << "Rule"
             << right << setw(4) << "Wt"
             << "   Status\n";
        cout << "  " << string(40, '-')
             << string(4, '-') << "  ------\n";

        int firedTotal = 0;
        for (const auto& r : results)
        {
            string status = r.fired ? "[FIRED]" : "[missed]";
            cout << "  " << left << setw(40) << r.name
                 << right << setw(4) << r.weight
                 << "  " << status << "\n";

            if (r.fired)
            {
                cout << "     -> " << r.description << "\n";
                firedTotal += r.weight;
            }
        }
          
        cout << "\n Rules fired: " << firedTotal
         << " / " << totalWeight() << " total weight\n";

    }

    // summary box

    cout << "\n  +------------------------------+\n";
    cout << "  | File:    " << left << setw(26) << f.filename << "|\n";
    cout << "  | AI Prob: " << fixed << setprecision(1)
         << score * 100 << "%"
         << string(24 - 6, ' ') << "|\n";
    cout << "  | Verdict: " << left << setw(26) << verdict(score) << "|\n";
    cout << "  +------------------------------+\n\n";

}

// main

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        cout << "\nUsage:\n"
             << "  ./rule_based_detector <file.cpp>\n"
             << "  ./rule_based_detector <file.cpp> --verbose\n\n";
        return 1;
    }

    string filepath = argv[1];
    bool verbose = false;
    for (int i = 2; i < argc; i++)
    {
        string a = argv[i];
        if (a == "--verbose" || a == "-v") verbose = true;
    }

    FeatureSet f = extractFeatures(filepath);
    auto results = evaluateRules(f);
    double score = computeScore(results);

    printReport(f, results, score, verbose);
    return 0;

}


