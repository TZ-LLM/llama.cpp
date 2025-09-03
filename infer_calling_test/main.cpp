#include "infer.h"
#include <iostream>

// define a static buffer to store the input and output
static char input[1024] = "";
static char output[1024] = "";

int main(int argc, char *argv[]) {
  // infinite loop to keep the program running
  while (true) {
    // get the input from the user
    std::cout << "Please input your question" << std::endl;
    std::cin.getline(input, 1024);

    // call the infer function
    int ret = infer(argc, argv, input, 1024, output, 1024);

    std::cerr << "infer returned with error code: " << ret << std::endl;
    std::cout << "output: " << output << std::endl;
  }
  return 0;
}
