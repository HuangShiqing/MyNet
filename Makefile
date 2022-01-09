CROSS_COMPILE =#arm-linux-gnueabi-#aarch64-linux-gnu-#指定交叉编译器
DEBUG = 1#指定当前为debug模式
MNN_DIR = /home/hsq/DeepLearning/clone/MNN
YAML_DIR = /home/hsq/DeepLearning/clone/yaml-cpp/

CC = $(CROSS_COMPILE)gcc#指定编译器
CXX = $(CROSS_COMPILE)g++#指定编译器
CCFLAGS = -Wall -fPIC
CXXFLAGS = -std=c++11 -Wall -fPIC
#指定头文件目录
CCFLAGS += -I$(YAML_DIR)/include -I./include/my_net -I./include/my_infer -I./include/ -I./3rd_party/stb_image -I$(MNN_DIR)/include
CXXFLAGS += -I$(YAML_DIR)/include -I./include/my_net -I./include/my_infer -I./include/ -I./3rd_party/stb_image -I$(MNN_DIR)/include
#指定库文件目录
LDFLAGS = -L$(MNN_DIR)/build -L$(YAML_DIR)/build
#指定库文件名称
LIBS = -lMNN -lstdc++ -lm -lyaml-cpp
#告诉makefile去哪里找依赖文件和目标文件
VPATH = src/my_net/:src/my_infer/:src/:example/
#最终生成的可执行文件名
CC_EXAMPLE = rfb320_cc_example
EXAMPLE = rfb320_example
DLIB = libMyNet.so
SLIB = libMyMet.a

#选择debug还是release
ifeq ($(DEBUG), 1)
CCFLAGS+=-O0 -g
CXXFLAGS+=-O0 -g
else
CCFLAGS+=-Ofast
CXXFLAGS+=-Ofast
endif

#存放.o文件的文件夹
OBJDIR = ./build/obj/
#中间过程所涉及的.o文件	
OBJ = mnn_infer.o base_infer.o rfb320.o base_net.o data_loader.o wrapper_rfb320.o  image.o 
OBJS = $(addprefix $(OBJDIR), $(OBJ))#添加路径
EXAMPLE_OBJ = rfb320_example.o
EXAMPLE_OBJS = $(addprefix $(OBJDIR), $(EXAMPLE_OBJ))#添加路径
CC_EXAMPLE_OBJ = rfb320_cc_example.o
CC_EXAMPLE_OBJS = $(addprefix $(OBJDIR), $(CC_EXAMPLE_OBJ))#添加路径

#指定需要完成的编译的对象
all: obj $(DLIB) $(SLIB) $(EXAMPLE) $(CC_EXAMPLE)

#将所有的.o文件链接成最终的可执行文件，需要库目录和库名，注意，OBJS要在LIBS之前。另外，如果要指定.o的生成路径，需要保证EXAMPLE的依赖项是含路径的
$(CC_EXAMPLE):$(CC_EXAMPLE_OBJS) $(SLIB)
		$(CC) $(CCFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS) $(SLIB)	
$(EXAMPLE):$(EXAMPLE_OBJS) $(SLIB)
		$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS) $(SLIB)
$(SLIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^
#这个LIBS一定要放在最后或者某个东西后面, 不然不会链接进动态库
$(DLIB): $(OBJS)
	$(CXX) $(CXXFLAGS) -shared  $^ -o $@ $(LDFLAGS) $(LIBS)
#这个不是静态模式，而是通配符，第一个%类似bash中的*
$(OBJDIR)%.o: %.c
		$(CC) -c $(CCFLAGS) $< -o $@
$(OBJDIR)%.o: %.cpp
		$(CXX) -c $(CXXFLAGS) $< -o $@

#用于生成存放.o文件的文件夹
obj:
	mkdir -p ./build/obj
.PHONY : 
	clean
clean :#删除生成的文件夹
	rm -rf ./build $(EXAMPLE) $(CC_EXAMPLE) $(DLIB) $(SLIB)