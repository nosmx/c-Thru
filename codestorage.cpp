#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <thread>
#include <chrono>
using namespace std;

int main()
{
    ofstream outputFile;
    outputFile.open("Code.txt");

    string code;

    cout << "Copy and paste your code here (press enter twice to finish): " << endl;
    while (true)
    {
        string line;
        getline(cin, line);
        if (line.empty())
        {
            break;
        }
        code += line + "\n";
    }

    cout << "Your code is now being saved to Code.txt..." << endl;
    this_thread::sleep_for(chrono::seconds(2)); // sim saving time
    cout << "Code has been saved to Code.txt" << endl;

    outputFile << code;
    outputFile.close();

    return 0;

}