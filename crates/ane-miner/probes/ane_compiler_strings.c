#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <mach-o/loader.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    const char *path = "/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler";
    if (dlopen(path, RTLD_NOW | RTLD_LOCAL) == NULL) {
        fprintf(stderr, "load failed: %s\n", dlerror());
        return 1;
    }
    for (uint32_t i = 0; i < _dyld_image_count(); ++i) {
        const char *name = _dyld_get_image_name(i);
        if (strstr(name, "/ANECompiler.framework/") == NULL) continue;
        const struct mach_header_64 *header = (const void *)_dyld_get_image_header(i);
        if (header->magic != MH_MAGIC_64) return 2;
        intptr_t slide = _dyld_get_image_vmaddr_slide(i);
        const struct load_command *command = (const void *)(header + 1);
        fprintf(stderr, "image=%s commands=%u\n", name, header->ncmds);
        for (uint32_t j = 0; j < header->ncmds; ++j) {
            if (command->cmd == LC_SEGMENT_64) {
                const struct segment_command_64 *segment = (const void *)command;
                const struct section_64 *sections = (const void *)(segment + 1);
                for (uint32_t k = 0; k < segment->nsects; ++k) {
                    const struct section_64 *section = &sections[k];
                    if ((section->flags & SECTION_TYPE) != S_CSTRING_LITERALS) continue;
                    fprintf(stderr, "section=%.16s size=%llu\n", section->sectname, (unsigned long long)section->size);
                    const char *bytes = (const void *)(section->addr + slide);
                    for (uint64_t offset = 0; offset < section->size;) {
                        size_t length = strnlen(bytes + offset, section->size - offset);
                        if (length > 0) printf("%.*s\n", (int)length, bytes + offset);
                        offset += length + 1;
                    }
                }
            }
            command = (const void *)((const char *)command + command->cmdsize);
        }
    }
    return 0;
}
