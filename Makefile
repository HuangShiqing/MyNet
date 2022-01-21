CROSS_COMPILE =#arm-linux-gnueabi-#aarch64-linux-gnu-#指定交叉编译器
DEBUG = 1#指定当前为debug模式

MNN = 1
MNN_DIR = /home/hsq/DeepLearning/clone/MNN

PADDLE_LITE = 0
TOOLCHAIN_DIR = /home/huangshiqing/toolchain/XTCL
PADDLE_LITE_BUILD_DIR = /data/huangshiqing/DeepLearning/code/Paddle-Lite/build.lite.x86

PADDLE = 0
PADDLE_DIR = /data/huangshiqing/DeepLearning/code/Paddle

YAML_DIR = /home/hsq/DeepLearning/clone/yaml-cpp

CC = $(CROSS_COMPILE)gcc#指定编译器
CXX = $(CROSS_COMPILE)g++#指定编译器
CCFLAGS = -Wall -fPIC
#指定头文件目录
CCFLAGS += -I$(YAML_DIR)/include \
		   -I./include/my_net \
		   -I./include/my_infer \
		   -I./include/ \
		   -I./3rd_party/stb_image
#指定库文件目录
LDFLAGS = -L$(YAML_DIR)/build
#指定库文件名称
LIBS = -lstdc++ -lm -lyaml-cpp
#告诉makefile去哪里找依赖文件和目标文件
VPATH = src/my_net/:src/my_infer/:src/:example/
#存放.o文件的文件夹
OBJDIR = ./build/obj/
#中间过程所涉及的.o文件
OBJ = base_infer.o base_net.o data_loader.o image.o compare.o
CXXFLAGS=${CCFLAGS} -std=c++11
OBJS = $(addprefix $(OBJDIR), $(OBJ))#添加路径

#最终生成的可执行文件名
EXES = 
DLIB = libMyNet.so
SLIB = libMyMet.a
#指定需要完成的编译的对象
all: obj $(DLIB) $(SLIB) EXAMPLE CC_EXAMPLE

#选择debug还是release
ifeq ($(DEBUG), 1)
CCFLAGS+=-O0 -g
else
CCFLAGS+=-Ofast
endif

ifeq ($(MNN), 1)
CCFLAGS+=-I$(MNN_DIR)/include
LDFLAGS+=-L$(MNN_DIR)/build
LIBS+=-lMNN
OBJ+=mnn_infer.o rfb320.o wrapper_rfb320.o
MNN_EXE=mnn_infer_example
EXES+=$(MNN_EXE)
EXE_OBJS=$(addprefix $(OBJDIR), $(MNN_EXE).o)#添加路径
$(MNN_EXE):$(EXE_OBJS) $(SLIB)
		$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS) $(SLIB)

RFB_EXE=rfb320_example
EXES+=$(RFB_EXE)
EXE_OBJS=$(addprefix $(OBJDIR), $(RFB_EXE).o)#添加路径
$(RFB_EXE):$(EXE_OBJS) $(SLIB)
		$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS) $(SLIB)

RFB_CC_EXE=rfb320_cc_example
EXES+=$(RFB_CC_EXE)
EXE_OBJS=$(addprefix $(OBJDIR), $(RFB_CC_EXE).o)#添加路径
$(RFB_CC_EXE):$(EXE_OBJS) $(SLIB)
		$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS) $(SLIB)		
endif

# PADDLE_LITE_BUILD_DIR should be ahead of PADDLE_DIR otherwise will cause namespace error
ifeq ($(PADDLE_LITE), 1)
CCFLAGS+=-I${PADDLE_LITE_BUILD_DIR}/inference_lite_lib/cxx/include/
LDFLAGS+=-L${TOOLCHAIN_DIR}/runtime/shlib -L${TOOLCHAIN_DIR}/shlib \
		 -L${PADDLE_LITE_BUILD_DIR}inference_lite_lib/third_party/mklml/lib/ \
		 -L${PADDLE_LITE_BUILD_DIR}/inference_lite_lib/cxx/lib
LIBS+=-lpaddle_full_api_shared -lxpuapi -lxpurt -liomp5
OBJS+=$(addprefix $(OBJDIR), paddle_lite_infer.o)
PADDLE_LITE_EXE=paddle_lite_infer_example
EXES+=$(PADDLE_LITE_EXE)
EXE_OBJS=$(addprefix $(OBJDIR), $(PADDLE_LITE_EXE).o)#添加路径
$(PADDLE_LITE_EXE):$(EXE_OBJS) $(SLIB)
		$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS) $(SLIB)
endif

ifeq ($(PADDLE), 1)
CCFLAGS+=-I${PADDLE_DIR}/paddle/fluid/inference/api \
		 -I${PADDLE_DIR}/paddle/fluid/framework/io \
		 -I${PADDLE_DIR}/build/third_party/install/gflags/include \
		 -I${PADDLE_DIR}/build/third_party/install/glog/include
LDFLAGS+=-L${PADDLE_DIR}/build/paddle/fluid/inference \
		 -L${PADDLE_DIR}/build/third_party/install/mklml/lib \
		 -L${PADDLE_DIR}/build/third_party/install/mkldnn/lib64 \
		 -L${PADDLE_DIR}/build/third_party/install/glog/lib \
		 -L${PADDLE_DIR}/build/third_party/install/gflags/lib
LIBS+=-lpaddle_inference -lgomp -liomp5 -lmklml_intel -ldnnl -lglog -lgflags
OBJ+=paddle_infer.o
PADDLE_EXE=paddle_infer_example
EXES+=$(PADDLE_EXE)
EXE_OBJS=$(addprefix $(OBJDIR), $(PADDLE_EXE).o)#添加路径
$(PADDLE_EXE):$(EXE_OBJS) $(SLIB)
		$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS) $(SLIB)

COMPARE_EXE=compare_example
EXES+=$(COMPARE_EXE)
EXE_OBJS=$(addprefix $(OBJDIR), $(COMPARE_EXE).o)#添加路径
$(COMPARE_EXE):$(EXE_OBJS) $(SLIB)
		$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS) $(SLIB)
endif

#将所有的.o文件链接成最终的可执行文件，需要库目录和库名，注意，OBJS要在LIBS之前。另外，如果要指定.o的生成路径，需要保证EXAMPLE的依赖项是含路径的
# $(EXAMPLE):$(EXAMPLE_OBJS) $(SLIB)
# 		$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS) $(SLIB)
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
EXAMPLE: $(EXES)
CC_EXAMPLE: $(CC_EXES)
#用于生成存放.o文件的文件夹
obj:
	mkdir -p ./build/obj
.PHONY : 
	clean
clean :#删除生成的文件夹
	rm -rf ./build $(EXES) $(CC_EXAMPLE) $(DLIB) $(SLIB)