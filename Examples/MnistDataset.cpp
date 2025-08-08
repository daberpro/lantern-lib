#include "../pch.h"
#include "../Dataset/Dataset.h"

/**
 * This is simple example of how to use lantern mnist dataset 
 * ==========================================================
 */

int main(){

    // Get mnist dataset class
    lantern::dataset::MnistDataset mnist;
    // print the train image and its label at index 10
    mnist.PrintTrainDataAt(10);

    // get train image as raw pointer  and its dimension
    auto buffer = mnist.GetTrainImageAt(10);
    auto dims = mnist.GetTrainImageDims(); // dimension size actuall has 3 values for image (number of images, width, height)
    af::array raw((*dims)[1],(*dims)[2], buffer); // then we create a new arrayfire array
    raw = raw.T(); // Transpose to correct the position of image shown

    // create a window
    af::Window app(500,500,"Mnist");
    while(!app.close()){
        app.image(raw);
    }

	return 0;
}