## 自定义算子向编译器声明语义
一个custom op调用底层kernel，编译器需要知道输出数量、shape/dtype/device/stride、哪些输入会修改、输出是否共享storage，以及反向规则。缺少这些信息，后端无法安全处理依赖和内存复用。

例如QK kernel直接写入KV cache。数学结果可能只返回Q，但cache mutation影响后面的attention。如果声明为无副作用纯函数，编译器可能错误地重排或消除调用。

## Fake 实现提供输出元数据
在不执行真实GPU计算时，编译器仍需要推导输出元信息。Fake implementation提供这一描述。它不应读取真实数据决定shape，也不能随意返回错误stride。Shape正确、layout错误，同样能使后端生成错误代码。

## Alias与输出新鲜度
某些custom-op契约要求返回新Tensor，不能隐式返回输入alias。项目compile接入将cache mutation显式声明，并在custom op外重建相应view。不要简单用“返回的是同一块cache，所以方便”替代注册系统允许的契约。

## 独立注册反向规则
推理custom op不必自动有训练支持。训练需要明确backward和saved state，检查mutation与梯度的交互。对带in-place/cache更新的操作，更不能默认为可微。

## 编译与性能的关系
把手写kernel包装成opaque custom op可能帮助捕获，却也可能形成fusion边界。fullgraph通过、零重编译说明编译接入可工作，不保证比已有compile快。要先验证图是否真的走预期kernel，再比较同一shape/backend的性能。

## 对照与测试
检查eager与compiled输出、不同shape的fake行为、mutation后的cache内容、alias、重复调用和错误入口拒绝。项目相关接口与版本固定，直接复制到新PyTorch版本应重新做回归，不能把私有API行为当公共保证。

这是DeepOpFuzz metadata错误案例与Qwen工程接入的另一个连接点：性能优化前需要把Tensor契约描述正确。
