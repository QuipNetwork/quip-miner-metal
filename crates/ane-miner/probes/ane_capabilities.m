#import <Foundation/Foundation.h>
#include <objc/runtime.h>
#include <dlfcn.h>
#include <stdio.h>

static void methods(Class cls, const char *kind) {
    unsigned count = 0;
    Method *list = class_copyMethodList(cls, &count);
    for (unsigned i = 0; i < count; ++i) {
        printf("  %s %s %s\n", kind, sel_getName(method_getName(list[i])), method_getTypeEncoding(list[i]));
    }
    free(list);
}

int main(void) {
    @autoreleasepool {
        if (dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW) == NULL) return 1;
        unsigned count = 0;
        Class *list = objc_copyClassList(&count);
        for (unsigned i = 0; i < count; ++i) {
            NSString *name = NSStringFromClass(list[i]);
            if (![name containsString:@"ANE"]) continue;
            printf("class=%s\n", name.UTF8String);
            unsigned propertyCount = 0;
            objc_property_t *properties = class_copyPropertyList(list[i], &propertyCount);
            for (unsigned p = 0; p < propertyCount; ++p) printf("  property %s %s\n", property_getName(properties[p]), property_getAttributes(properties[p]));
            free(properties);
            methods(list[i], "instance");
            methods(object_getClass(list[i]), "class");
        }
        free(list);
    }
    return 0;
}
