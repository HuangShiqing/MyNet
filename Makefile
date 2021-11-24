CROSS_COMPILE =#arm-linux-gnueabi-#aarch64-linux-gnu-#指定交叉编译器
DEBUG = 1#指定当前为debug模式
MNN_DIR = /home/hsq/DeepLearning/clone/MNN

CXX = $(CROSS_COMPILE)g++#指定编译器
CXXFLAGS = -std=c++11 -Wall -fPIC
#指定头文件目录
CXXFLAGS += -I./include/ -I./include/my_module/ -I./3rd_party/stb_image -I$(MNN_DIR)/include
#指定库文件目录
LDFLAGS = -L$(MNN_DIR)/build
#指定库文件名称
LIBS = -lMNN
#告诉makefile去哪里找依赖文件和目标文件
VPATH = ./src/:./src/my_module/:./example/
#最终生成的可执行文件名
EXAMPLE = rfb320_example
DLIB = libMyModule.so
SLIB = libMyModule.a

#选择debug还是release
ifeq ($(DEBUG), 1)
CXXFLAGS+=-O0 -g
else
CXXFLAGS+=-Ofast
endif

#存放.o文件的文件夹
OBJDIR = ./build/obj/
#中间过程所涉及的.o文件	
OBJ = rfb320.o base_module.o image.o my_net.o my_net_mnn.o 
OBJS = $(addprefix $(OBJDIR), $(OBJ))#添加路径
EXAMPLE_OBJ = rfb320_example.o
EXAMPLE_OBJS = $(addprefix $(OBJDIR), $(EXAMPLE_OBJ))#添加路径
#指定需要完成的编译的对象
all: obj $(DLIB) $(SLIB) $(EXAMPLE)

#将所有的.o文件链接成最终的可执行文件，需要库目录和库名，注意，OBJS要在LIBS之前。另外，如果要指定.o的生成路径，需要保证EXAMPLE的依赖项是含路径的
$(EXAMPLE):$(EXAMPLE_OBJS) $(SLIB)
		$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS) $(SLIB)
$(SLIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^
#这个LIBS一定要放在最后或者某个东西后面, 不然不会链接进动态库
$(DLIB): $(OBJS)
	$(CXX) $(CXXFLAGS) -shared  $^ -o $@ $(LDFLAGS) $(LIBS)
#这个不是静态模式，而是通配符，第一个%类似bash中的*
$(OBJDIR)%.o: %.cpp
		$(CXX) -c $(CXXFLAGS) $< -o $@

#用于生成存放.o文件的文件夹
obj:
		mkdir -p ./build/obj
.PHONY : clean
clean :#删除生成的文件夹
		rm -rf ./build $(EXAMPLE)