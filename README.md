# C-Thru
Detecting AI-Generated Introductory C++ Programs Through Static Code Analysis and Supervised Logistic Regression


As large language models increasingly generate syntactically correct code, distinguishing AI-generated from human-written programs has become a challenge in computer science education. This project investigates whether a machine learning model, logistic regression, can improve detection accuracy compared to a rule-based analysis system for introductory C++ programs. I will build a labeled collection of student-written and AI-generated introductory C++ programs and examine measurable patterns such as word choice, commenting style, structural organization, and overall program complexity. A rule-based scoring system will be implemented using these features and compared against a trained logistic regression model using the same dataset. I will measure how accurately each method identifies AI-generated code, how often it makes mistakes, and how consistently it performs across different types of assignments. I expect the comparison to reveal whether statistical models capture subtle structural patterns that rule-based systems may overlook, offering insight into how AI tools differ from novice programmers.


Run Features Extractor: 
Compile:
 *   g++ -std=c++17 -O2 feature_extractor.cpp features.cpp -o feature_extractor
 *
 * Usage:
 *   ./feature_extractor <file.cpp> [label]           single file
 *   ./feature_extractor <file.cpp> [label] --csv=out.csv
 *   -- label: 1 = AI-generated, 0 = human-written (omit if unknown)
 */
