#include "pch.h"
#include "AutoGradient/ReverseMode.h"
#include "Logging.h"
#include "Vector.h"
#include <arrayfire.h>

using namespace latern;

int main(){	

	// af::info();

	// af::array rn1 = af::randn(3,3,3);
	// af::array a1(2);
	// af::array i1 = af::identity(3,3);
	// af::array c1 = af::constant(1.0f, 4);
	// af_print(a1);
	// af_print(i1);
	// af_print(c1);
	// af_print(rn1);
	// af_print(af::where(rn1 > 0));

	// const uint32_t width = 400, height = 400;
	// af::Window window(width,height,"Test window");

	// // af::array img = af::constant(0,width,height);
	// // af::array noise = af::randu(width,height);
	// // img(noise > 0.5) = 1.0f;

	// af::array range(af::seq(-af::Pi,af::Pi, 0.01f));
	// af::array sin(af::sin(range));

	// do{
	// 	window.plot(range,sin);
	// }while(!window.close());

	::utility::Vector<std::string> v1;
	v1.push_back("Apel");
	for(std::string& v : v1){
		std::cout << v << "\n";
	}

	Node c1(2.5,"c1");
	Node c2(3.4,"c2");
	Node c3 = c1 * c2;
	Node c4 = c1 + c2;
	Node c5 = c3 / c4;
	Node c6 = c1 * c5;
	c3.SetLabel("c3");
	c4.SetLabel("c4");
	c5.SetLabel("c5");
	c6.SetLabel("c6");
	ReverseModeAD(c6);
	
	std::cout << std::string(50,'=') << "\n";
	print(c6);
	print(c5);
	print(c4);
	print(c3);
	print(c2);
	print(c1);
	std::cout << std::string(50,'=');

    // Node x(2.5);
    // Node sg = Sigmoid(x);
    // Node sg2 = Sigmoid(sg);

    // ReverseModeAD(sg2);
	// std::cout << std::string(30,'=') << "\n";
	// print(sg2,"Sigmoid 2");
	// print(sg,"Sigmoid 1");
	// print(x,"x");
	// std::cout << std::string(30,'=') << "\n\n";

    // Node a(2.0);
	// Node b(3.0);
	// Node c(4.0);
	// Node x1 = a + b;
	// Node x2 = x1 * c;
	
	// ReverseModeAD(x2);
	// std::cout << std::string(30,'=') << "\n";
	// print(x2,"x2");
	// print(x1,"x1");
	// print(a,"a");
	// print(b,"b");
	// print(c,"c");
	// std::cout << std::string(30,'=') << "\n\n";
	
    return EXIT_SUCCESS;
}