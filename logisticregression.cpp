#include "features.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath> // for functions like exp, sqrt
#include <algorithm>
#include <numeric>
#include <random> // used for shuffling dataset 
#include <functional> // used to pass functions the same way as variables
using namespace std;

// model file paths

static const string MODEL_FILE = "lr_weights.txt"; 
static const string SCALER_FILE = "lr_scaler.txt"; 

// math helpers

static double sigmoid(double z)
{
    return 1.0 / (1.0 + exp(-z));
}

static double dot(const vector<double>& a, const vector<double>& b)
{
    double s = 0.0;
    for (size_t i = 0; i < a.size(); i++)
    {
        s += a[i] * b[i];
    }
    return s;
}

// StandardScaler

struct Scaler
{
    vector<double> mean;
    vector<double> stddev;

    // fit scaler on training data

    void fit(const vector<vector<double>>& X)
    {
        int n = (int)X.size();
        int m = (int)X[0].size();
        mean.assign(m, 0.0);
        stddev.assign(m, 1.0);

        for (const auto& row : X)
        {
            for (int j = 0; j < m; j++)
            {
                mean[j] += row[j];
            }
        }
        for (auto& v : mean) v /= n;

        vector<double> var(m, 0.0);
        for (const auto& row : X)
        {
            for (int j = 0; j < m; j++)
            {
                double d = row[j] - mean[j];
                var[j] += d * d;
            }
        }
        for (int j = 0; j < m; j++)
        {
            stddev[j] = (var[j] / n > 1e-9) ? sqrt(var[j] / n) : 1.0;
        }
    }

    // transform a whole dataset using fitted parameters
    vector<vector<double>> transform(const vector<vector<double>>& X) const
    {
    vector<vector<double>> out = X;
    for (auto& row : out)
    {
        for (size_t j = 0; j < row.size(); j++)
        {
            row[j] = (row[j] - mean[j]) / stddev[j];
        }
    }
    return out;
}

    // transform a single sample
    vector<double> transformOne(const vector<double>& x) const
    {
        vector<double> out = x;
        for (size_t j = 0; j < out.size(); j++)
        {
            out[j] = (out[j] - mean[j]) / stddev[j];
        }

        return out;

    }

    bool save(const string& path) const
    {
        ofstream f(path);
        if (!f) return false;
        f << mean.size() << "\n";
        for (double v : mean) f << setprecision(10) << v << "\n";
        for (double v : stddev) f << setprecision(10) << v << "\n";
        return true;
    }

    bool load(const string& path)
    {
        ifstream f(path);
        if (!f) return false;
        int m; f >> m;
        mean.resize(m); stddev.resize(m);
        for (double& v : mean) f >> v;
        for (double& v : stddev) f >> v;
        return true;
    }
};

//logisitc regression model
struct LogisticRegression
{
    vector<double> weights;
    int numFeatures = 0;
    double learningRate = 0.1;
    int epochs = 2000;

    //predict probability that sample x is AI (class 1)
    double predictProb(const vector<double>& x) const
    {
        double z = weights[0];
        for (int j = 0; j < numFeatures; j++) 
        {
            z += weights[j + 1] * x[j];
        }
        return sigmoid(z);
    }

    int predict(const vector<double>& x) const
    {
        return predictProb(x) >= 0.5 ? 1 : 0;
    }

    //train using batch gradient descent
    void train(const vector<vector<double>>& X, const vector<int>& y)
    {
        int n = (int)X.size();
        numFeatures = (int)X[0].size();
        weights.assign(numFeatures + 1, 0.0);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            //compute gradients
            vector<double> grad(numFeatures + 1, 0.0);
            double loss = 0.0;
            for (int i = 0; i < n; i++)
            {
                double prob = predictProb(X[i]);
                double error = prob - double(y[i]);
                //cross-entropy loss
                double yi = double(y[i]);
                loss -= yi * log(prob + 1e-12);
                //gradient accumulation
                grad[0] += error; // bias gradient
                for (int j = 0; j < numFeatures; j++)
                {
                    grad[j + 1] += error * X[i][j];
                }
            }
            // update weights
            for(int j = 0; j <= numFeatures; j++)
            {
                weights[j] -= learningRate * grad[j] / n;
            }

            // print progress every 500 epochs
            if((epoch + 1) % 500 == 0)
            {
                cout << "  Epoch " << setw(4) << (epoch + 1) << "  Loss: " << fixed << setprecision(4) << loss / n << "\n";
            }
        }
        
    }

    bool save(const string& path) const
    {
        ofstream f(path);
        if (!f) return false;
        f << numFeatures << "\n";
        for (double w : weights)
        {
            f << setprecision(10) << w << "\n";
        }

        return true;
    }

    bool load(const string& path)
    {
        ifstream f(path);
        if (!f) return false;
        f >> numFeatures;
        weights.resize(numFeatures + 1);
        for (double& w : weights) f >> w;
        return true;
    }
};

// evaluation metrics

struct Metrics
{
    double accuracy = 0;
    double precision = 0;
    double recall = 0;
    double f1 = 0;
    int tp = 0;
    int tn = 0;
    int fp = 0;
    int fn = 0;
};

static Metrics evaluate(const vector<int>& predictions, const vector<int>& labels)
{
    Metrics m;
    int n = (int)labels.size();
    for(int i = 0; i < n; i++)
    {
        if (labels[i] == 1 && predictions[i] == 1) m.tp++;
        else if (labels[i] == 0 && predictions[i] == 0) m.tn++;
        else if (labels[i] == 0 && predictions[i] == 1) m.fp++;
        else m.fn++;
    }

    m.accuracy = double(m.tp + m.tn) / n;
    m.precision = (m.tp + m.fp > 0) ? (double)m.tp / (m.tp + m.fp) : 0;
    m.recall = (m.tp + m.fn > 0) ? (double)m.tp / (m.tp + m.fn) : 0;
    m.f1 = (m.precision + m.recall > 0) ? 2.0 * m.precision * m.recall / (m.precision + m.recall) : 0;
    return m;
}

static void printMetrics(const Metrics& m, const string& label = "")
{
    string LINE(55, '-');
    if (!label.empty()) cout << "\n" << label << "\n" << LINE << "\n";

    cout << "  Accuracy: " << fixed << setprecision(1) << m.accuracy * 100 << "%\n";
    cout << "  Precision: " << fixed << setprecision(1) << m.precision * 100 << "% (of files called AI, how many really are)\n";
    cout << "  Recall: " << fixed << setprecision(1) << m.recall * 100 << "% (of all real AI files, how many we caught)\n";
    cout << "  F1 Score: " << fixed << setprecision(3) << m.f1 << "\n";
    cout << "\n Confusion Matrix:\n";
    cout << " Pred \\ Human | AI\n";
    cout << "  Actual Human " << setw(6) << m.tn << "   " << setw(6) << m.fp << "\n";
    cout << "  Actual AI.   " << setw(6) << m.fn << "   " << setw(6) << m.tp << "\n";
}

// train/test split utility

static void splitDataset(const vector<vector<double>>& X, const vector<int>& y, double testRatio, vector<vector<double>>& Xtrain, vector<int>& ytrain, vector<vector<double>>& Xtest, vector<int>& ytest)
{
    int n = (int)X.size();
    vector<int> idx(n);
    iota(idx.begin(), idx.end(), 0 );

    mt19937 rng(42);
    shuffle(idx.begin(), idx.end(), rng);

    int testSize = max(1, (int)(n * testRatio));
    for (int i = 0; i < testSize; i++)
    {
        Xtest.push_back(X[idx[i]]);
        ytest.push_back(y[idx[i]]);
    }

    for(int i = testSize; i < n; i++)
    {
        Xtrain.push_back(X[idx[i]]);
        ytrain.push_back(y[idx[i]]);
    }
}

// feature importance : print top weighted features


static void printFeatureImportance(const LogisticRegression& model)
{
    auto names = featureNames();
    vector<pair<double, string>> ranked;
    for(int j = 0; j < model.numFeatures; j++)
    {
        ranked.push_back({model.weights[j+1], names[j] });
    }

    sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b)
    {
        return abs(a.first) > abs(b.first);
    });

    cout << "\n Top 10 Most Productive Features:\n";
    cout << "  " << left << setw(35) << "Feature" << right << setw(12) << "Coefficient" << "   Direction\n";
    cout << "  " << string(35, '-') << "   " << string(12, '-') << "   " << string(9, '-') << "\n";
    for (int i = 0; i < min(10, (int)ranked.size()); i++)
    {
        string dir = ranked[i].first > 0 ? "-> AI" : "-> Human";
        cout << "  " << left << setw(35) << ranked[i].second << right << setw(12) << fixed << setprecision(4) << ranked[i].first << "  " << dir << "\n";
    }
}

// ruled based scoring helper 


static double ruleBasedScore(const FeatureSet& f)
{
    int fired = 0;
    int total = 85;

    auto check = [&](bool cond, int w) {if(cond) fired += w;};

    check(f.comment_density > 0.25, 12);
    check(f.average_comment_length > 40.0, 8);
    check(f.block_comment_count >= 1, 5);
    check(f.average_identifier_length > 8.0, 10);
    check(f.unique_identifier_count > 20, 6);
    check(f.indentation_consistency > 0.90, 10);
    check(f.magic_number_count <= 2, 5);
    //check(f.open_brace_count == f.close_brace_count, 4);
    check(f.if_count > 0 && (double)f.else_count / f.if_count >= 0.75,7);
    check(f.control_count >= 3 && f.control_count <= 15, 5);
    //check(f.using_namespace_std == 1, 6);
    //check(f.endl_count >= f.cout_count, 5);
    check(f.non_empty_lines >= 25 && f.non_empty_lines <= 90, 5);
    check(f.total_lines > 0 && (double)f.blank_lines_count / f.total_lines <= 0.15, 4);
    check(f.max_nesting_depth <= 3, 4);
    check(f.operator_diversity < 8,4);

    return (double)fired / total;

}


// command: train

static int cmdTrain(const string& csvPath)
{
    vector<vector<double>> X;
    vector<int> y;
    vector<string> filenames;

    if (!loadDataset(csvPath, X, y, filenames)) return 1;

    if ((int)X.size() < 10)
    {
        cout << "[!] Need at least 10 labeled samples to train a model.  "
        << "Add more .cpp files to your dataset.\n";

        return 1;
    }

    int aiCount = 0;
    int humanCount = 0;
    for (int lbl : y) {if (lbl==1) aiCount++; else humanCount++;}
    cout << "   AI samples: " << aiCount << "\n";
    cout << "   Human samples: " << humanCount << "\n";

    // train / test split (80/20)

    vector<vector<double>> Xtrain, Xtest;
    vector<int> ytrain, ytest;
    splitDataset(X, y, 0.20, Xtrain, ytrain, Xtest, ytest);

    cout << "  Training on " << Xtrain.size() << " samples,  " << "testing on " << Xtest.size() << "samples\n\n";


    // scale

    Scaler scaler;
    scaler.fit(Xtrain);
    auto XtrainS = scaler.transform(Xtrain);
    auto XtestS = scaler.transform(Xtest);

    // Train
    cout << "Training logistic regression (gradient descent)...\n";
    LogisticRegression model;
    model.learningRate = 0.05;
    model.epochs = 2000;
    model.train(XtrainS, ytrain);

    // evaluate on test set
    vector<int> testPreds;
    for (const auto& x : XtestS)
    {
        testPreds.push_back(model.predict(x));
    }
    Metrics m = evaluate(testPreds, ytest);

    cout << "\n" << string(55, '=') << "\n";
    cout << "   Logistic Regression - Results\n";
    cout << string(55, '=') << "\n";
    printMetrics(m, "Test Set Performance");

    printFeatureImportance(model);

    // save model and scaler

    model.save(MODEL_FILE);
    scaler.save(SCALER_FILE);
    cout << "\n [:)] Model saved -> " << MODEL_FILE << "\n";
    cout << " [:)] Scaler saved -> " << SCALER_FILE << "\n";

    return 0;


}

// command: predict

static int cmdPredict(const string& filepath) 
{
    LogisticRegression model;
    Scaler scaler;

    if (!model.load(MODEL_FILE) || !scaler.load(SCALER_FILE))
    {
        cout << "[!] No trained model found.\n";
        cout << "   Run ./logisticregression train features.csv\n";

        return 1;
    }

    FeatureSet f = extractFeatures(filepath);
    auto x = featureVector(f);
    auto xs = scaler.transformOne(x);
    double prob = model.predictProb(xs);

    string verd = (prob >= 0.7) ? "Likely AI-generated" : (prob >= 0.45) ? "Uncertain / Mixed" : "Likely Human Written";

    cout << "\n +---------------------------------------------+\n";
    cout << "  | Logistic Regression Prediction              |\n";
    cout << "  | File:    " << left << setw(30) << f.filename << "|\n";
    cout << "  | AI Prob: " << fixed << setprecision(1) << prob * 100 << "%" << string (34, ' ') << "|\n";
    cout << "  | Verdict: " << left << setw(37) << verd << "|\n";
    cout << " +---------------------------------------------+\n\n";
    return 0;

}

// command: compare (rule based vs logistic regression using full dataset)

static int cmdCompare(const string& csvPath)
{
    vector<vector<double>> X;
    vector<int> y;
    vector<string> filenames;

    if (!loadDataset(csvPath, X, y, filenames)) return 1;

    LogisticRegression model;
    Scaler scaler;
    if (!model.load(MODEL_FILE) || !scaler.load(SCALER_FILE))
    {
        cout << "[!] Train the model first:\n";
        cout << "   ./logisticregression train " << csvPath << "\n";
        return 1;
    }

    // logistic regression predictions on full dataset
    auto Xs = scaler.transform(X);
    vector<int> lrPreds;
    for(const auto& x : Xs) lrPreds.push_back(model.predict(x));

    // rule based preidctions on full dataset
    // re read og CSV to reoncstruct FeatureSet for rule scoring
    vector<int> rbPreds;
    {
        ifstream in(csvPath);
        string headerLine;
        getline(in, headerLine);

        vector<string> cols;
        istringstream hss(headerLine);
        string col;
        while (getline(hss, col, ',')) cols.push_back(col);

        auto findCol = [&](const string& name)
        {
            for(int i = 0; i < (int)cols.size(); i++)
            {
                if (cols[i] == name) return i;
            }

            return -1;
        };

        // Lambda to get a double value from a CSV row by column name
        auto get = [&](const vector<string>& fields, const string& name) -> double 
        {
            int idx = findCol(name);
            if (idx < 0 || idx >= (int)fields.size()) return 0.0;
            try { return stod(fields[idx]); } catch(...) { return 0.0; }
        };

        string line;
        while (getline(in, line))
        {
            if(line.empty()) continue;
            vector<string> fields;
            istringstream ss(line);
            string field;
            while(getline(ss, field, ',')) fields.push_back(field);

            

            int lbl = -1;
            int labelIdx = findCol("label");
            if(labelIdx >= 0 && labelIdx < (int)fields.size())
            {
                try {lbl = stoi(fields[labelIdx]); } catch(...) {}
            }
            if(lbl < 0) continue;

            // reconstruct a FeatureSet from the row for rule scoring
            FeatureSet f{};
            f.total_lines = (int)get(fields, "total_lines");
            f.non_empty_lines = (int)get(fields, "non_empty_lines");
            f.blank_lines_count = (int)get(fields, "blank_lines_count");
            f.max_line_length = (int)get(fields, "max_line_length");
            f.average_line_length = get(fields, "average_line_length");
            f.open_brace_count = (int)get(fields, "open_brace_count");
            f.close_brace_count = (int)get(fields, "close_brace_count");
            f.line_comment_count = (int)get(fields, "line_comment_count");
            f.block_comment_count = (int)get(fields, "block_comment_count");
            f.total_comment_lines = (int)get(fields, "total_comment_lines");
            f.comment_density = get(fields, "comment_density");
            f.average_comment_length = get(fields, "average_comment_length");
            f.if_count = (int)get(fields, "if_count");
            f.else_count = (int)get(fields, "else_count");
            f.for_count = (int)get(fields, "for_count");
            f.while_count = (int)get(fields, "while_count");
            f.switch_count = (int)get(fields, "switch_count");
            f.return_count = (int)get(fields, "return_count");
            f.control_count = (int)get(fields, "control_total");
            f.include_count = (int)get(fields, "include_count");
            f.using_namespace_std = (int)get(fields, "using_namespace_std");
            f.std_prefix_count = (int)get(fields, "std_prefix_count");
            f.cout_count = (int)get(fields, "cout_count");
            f.cin_count = (int)get(fields, "cin_count");
            f.endl_count = (int)get(fields, "endl_count");
            f.average_identifier_length = get(fields, "average_identifier_length");
            f.unique_identifier_count = (int)get(fields, "unique_identifier_count");
            f.indentation_consistency = get(fields, "indentation_consistency");
            f.magic_number_count = (int)get(fields, "magic_number_count");
            f.max_nesting_depth = (int)get(fields, "max_nesting_depth");
            f.function_count = (int)get(fields, "function_count");
            f.operator_diversity = (int)get(fields, "operator_diversity");

            double score = ruleBasedScore(f);
            rbPreds.push_back(score >= 0.5 ? 1 : 0);
            
            
        }
    }

    // trim to same size if mistmatch 
    int n = min({(int)y.size(), (int)lrPreds.size(), (int)rbPreds.size()});
    y.resize(n); lrPreds.resize(n); rbPreds.resize(n);

    Metrics rbM = evaluate(rbPreds, y);
    Metrics lrM = evaluate(lrPreds, y);

    // print side by side comparison
    cout << "\n" << string(60, '=') << "\n";
    cout << "  Model Comparison -  Rule Based vs. Logistic Regression\n";
    cout << string(60, '=') << "\n\n";

    cout << "  " << left << setw(32) << "Method" << right << setw(10) << "Accuracy" << setw(12) << "F1 Score" << "\n";
    cout << "  " << string(32, '-') << string(10, '-') << string(12, '-') << "\n";

    string rbWin = (rbM.accuracy > lrM.accuracy) ? "  <- WINS" : "";
    string lrWin = (lrM.accuracy > rbM.accuracy) ? "  <- WINS" : "";

    cout << "  " << left << setw(32) << "Rule Based Static Analysis" << right << setw(9) << fixed << setprecision(1) << rbM.accuracy * 100 << "%" << setw(11) << setprecision(3) << rbM.f1 << rbWin << "\n";
    cout << "  " << left << setw(32) << "Logistic Regression Model" << right << setw(9) << fixed << setprecision(1) << lrM.accuracy * 100 << "%" << setw(11) << setprecision(3) << lrM.f1 << lrWin << "\n";

    printMetrics(rbM, "Rule Based Detailed Metrics");
    printMetrics(lrM, "Logistic Regression Detailed Metrics");

    // per file breakdown
    cout << "\n Per-File Breakdown:\n";
    cout << "  " << left << setw(30) << "Filename" << setw(8) << "True" << setw(10) << "RuleBsd" << setw(8) << "LogReg" << "Notes\n";
    cout << "  " << string(64, '-') << "\n";
    for (int i = 0; i < n; i++)
    {
        string fn = (i < (int)filenames.size()) ? filenames[i] : "?";
        if(fn.size() > 28) fn = fn.substr(fn.size()-28);
        string trueStr = y[i] ? "AI" : "Human";
        string rbStr = rbPreds[i] ? "AI" : "Human";
        string lrStr = lrPreds[i] ? "AI" : "Human";
        string note = "";

        if (rbPreds[i] != y[i] && lrPreds[i] != y[i]) note = "<- both wrong";
        else if (rbPreds[i] != y[i]) note = "<- rule based wrong";
        else if (lrPreds[i] != y[i]) note = "<- logistic regression wrong";
        cout << "  " << left << setw(30) << fn << setw(8) << trueStr << setw(10) << rbStr << setw(8) << lrStr << note << "\n";
    }
    cout << "\n";
    return 0;
}

// main function

static void printUsage()
{
    cout << "\nUsage:\n";
    cout << "  ./logisticregression train <features.csv>\n";
    cout << "  ./logisticregression predict <file.cpp>\n";
    cout << "  ./logisticregression compare <features.csv>\n\n";
    cout << "Workflow:\n";
    cout << "  1. Build features.csv with feature_extractor\n";
    cout << "  2. Train:   ./logisticregression train features.csv\n";
    cout << "  3. Predict: ./logisticregression predict newfile.cpp\n";
    cout << "  4. Compare: ./logisticregression compare features.csv\n\n";
}

int main(int argc, char* argv[])
{
    if (argc < 3) {printUsage(); return 1;}

    string cmd = argv[1];
    string target = argv[2];

    if (cmd == "train") return cmdTrain(target);
    else if (cmd == "predict") return cmdPredict(target);
    else if (cmd == "compare") return cmdCompare(target);
    else
    {
        cout << "[!] Unknown command: " << cmd << "\n";
        printUsage();
        return 1; 
    }
}


