// Bead quip-miner-metal-kyk, experiment 2 reconnaissance: what do the
// weight-symbol binding classes want?
//
// The runtime exposes _ANEWeight (weightSymbol, weightURL, SHACode),
// _ANEProcedureData (procedureDataWithSymbol:weightArray:), and
// _ANEModelInstanceParameters (withProcedureData:procedureArray:), and
// _ANEClient takes loadModelNewInstance:options:modelInstParams:qos:error:.
// Together those are the shape the notes describe for per-adapter weight
// files bound to an already-compiled base model.
//
// This probe does not load anything. It prints the runtime's own naming
// strings and checks that the three objects can be constructed from an
// unsigned binary, so experiment 2 starts from facts rather than guesses
// about symbol names. Production bridge entry points are unchanged.
#import <Foundation/Foundation.h>
#include <dlfcn.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <stdio.h>

static void printString(Class cls, SEL selector, const char *label) {
    if (cls == nil || ![cls respondsToSelector:selector]) {
        printf("%s=<unavailable>\n", label);
        return;
    }
    id (*call)(id, SEL) = (id (*)(id, SEL))objc_msgSend;
    id value = call(cls, selector);
    printf("%s=%s\n", label, [value isKindOfClass:NSString.class] ? [value UTF8String] : "<not a string>");
}

int main(void) {
    setbuf(stdout, NULL);
    @autoreleasepool {
        if (dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW) == NULL) {
            fprintf(stderr, "ANE framework dlopen failed\n");
            return 1;
        }

        Class strings = NSClassFromString(@"_ANEStrings");
        printString(strings, NSSelectorFromString(@"defaultWeightFileName"), "default_weight_file_name");
        printString(strings, NSSelectorFromString(@"adapterWeightsAccessEntitlement"), "adapter_entitlement");
        printString(strings, NSSelectorFromString(@"adapterWeightsAccessEntitlementBypassBootArg"), "adapter_bypass_boot_arg");

        // Can an unsigned binary build the three objects at all? A failure
        // here would end experiment 2 before it needs a device.
        Class weightClass = NSClassFromString(@"_ANEWeight");
        Class procedureClass = NSClassFromString(@"_ANEProcedureData");
        Class parametersClass = NSClassFromString(@"_ANEModelInstanceParameters");
        printf("classes: weight=%d procedure_data=%d instance_parameters=%d\n",
            weightClass != nil, procedureClass != nil, parametersClass != nil);

        NSURL *url = [NSURL fileURLWithPath:@"/tmp/quip-ane-weight-probe.bin"];
        id (*weightCall)(id, SEL, id, id) = (id (*)(id, SEL, id, id))objc_msgSend;
        id weight = weightCall(weightClass, NSSelectorFromString(@"weightWithSymbolAndURL:weightURL:"),
            @"weight_data.bin", url);
        printf("weight_built=%d symbol=%s url=%s\n", weight != nil,
            weight != nil ? [[weight valueForKey:@"weightSymbol"] UTF8String] : "<nil>",
            weight != nil ? [[[weight valueForKey:@"weightURL"] path] UTF8String] : "<nil>");
        if (weight == nil) return 1;

        id procedure = weightCall(procedureClass, NSSelectorFromString(@"procedureDataWithSymbol:weightArray:"),
            @"main", @[weight]);
        printf("procedure_data_built=%d\n", procedure != nil);
        if (procedure == nil) return 1;

        id parameters = weightCall(parametersClass, NSSelectorFromString(@"withProcedureData:procedureArray:"),
            procedure, @[procedure]);
        printf("instance_parameters_built=%d\n", parameters != nil);

        // What does the client class look like, and can one be made without
        // going through a model?
        Class clientClass = NSClassFromString(@"_ANEClient");
        printf("client_class=%d responds_to_load_new_instance=%d\n", clientClass != nil,
            clientClass != nil && [clientClass instancesRespondToSelector:
                NSSelectorFromString(@"loadModelNewInstance:options:modelInstParams:qos:error:")]);
        return 0;
    }
}
