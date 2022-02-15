#include <iostream>
#include <chrono>
#include <algorithm>
#include <fstream>

#include "cifar_reader/cifar10_reader.hpp"
#ifdef BINARY
#define INPUT_FEATURE char
#include "net.hpp"
#elif INT16
#define INPUT_FEATURE int
#include "net.hpp"
#else
#define INPUT_FEATURE float
#include "net.hpp"
#endif
using namespace std;
auto benchmark(bool verbose = false) {
#if defined BINARY || defined INT16
    int output[10] = {0};
#else
    float output[10] = {0};
#endif
    int factor = 1;
    int matches = 0;
    auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    auto test_images = dataset.test_images;
    auto test_labels = dataset.test_labels;
    unsigned char img[32][32][3];

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i+=factor) {
      for (int j = 0; j < test_images[i].size(); j++) {
        int d3 = j / 1024;
        int minus = j % 1024;
        int d2 = minus % 32;
        int d1 = minus / 32;
        img[d1][d2][d3] = static_cast<unsigned char>(test_images[i][j]);
      }
      std::fill(output, output+10, 0);
      int label = static_cast<int>(test_labels[i]);

        // display img array
        ofstream g("original_img.out");
        // for(int b=0;b<BATCH_SIZE;b++){
            for(int c=0;c<3;c++){
                g<<"batch: 0"<<", label: "<<label<<", channel: "<<c<<endl;
                // printf("batch: %d, label: %d, channel: %d\n",b,label,c);
                for (int i = 0; i < 32; i++)
                {
                    for (int j = 0; j < 32; j++)
                    {
                        g<<int(img[i][j][c])<<" ";
                        // printf("%d ", img[i][j][c]);
                    }
                    g<<endl;
                    // printf("\n");
                }
                g<<endl<<endl;
                // printf("\n\n");
            }
        // }

      predict_NeuralNet(img, output);
      
      cout<<i<<"(outp): ";
        for(int i=0;i<10;i++){
            cout<<output[i]<<", ";
        }
        printf("\n");
      float max = output[0];
      int argmax = 0;
      for (int j = 1; j < 10; j++) {
          if (output[j] > max) {
              max = output[j];
              argmax = j;
          }
      }
        std::cout<<i<<": "<<argmax<<"=="<<label<<std::endl;
      if (argmax == label) {
          matches++;
      }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto runtime = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()) / (test_images.size()/factor);
    float accuracy = static_cast<float>(matches) / (test_images.size()/factor) * 100.f;
    return std::make_pair(accuracy, runtime);
  }

int main() {
    // benchmark();
    auto results = benchmark();
    std::cout << "Accuracy: " << results.first << " %" << std::endl;
    std::cout << "Latency: " << results.second << " [ms/elem]" << std::endl;

    return 0;
}
