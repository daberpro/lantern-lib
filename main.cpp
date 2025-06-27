#include "pch.h"
#include "Headers/Logging.h"
#include "FeedForwardNetwork/FeedForwardNetwork.h"
#include "Headers/File.h"
#include "Dataset/Dataset.h"

int main(){

    lantern::dataset::MnistDataset mnist;
    mnist.PrintTrainDataAt(10);

    auto buffer = mnist.GetTrainImageAt(10);
    auto* dims = mnist.GetTrainImageDims();
    af::array raw((*dims)[1],(*dims)[2], buffer);
    raw = raw.T();

    af::Window app(500,500,"Mnist");
    while(!app.close()){
        app.image(raw);
    }

	return 0;
}