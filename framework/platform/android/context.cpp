//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "android/context.hpp"

#include <jni.h>

extern "C" {
JNIEXPORT void JNICALL
Java_com_khronos_vulkan_1samples_SampleLauncherActivity_initFilePath(JNIEnv *env, jobject thiz, jstring external_dir, jstring temp_dir) {
    const char *external_dir_cstr = env->GetStringUTFChars(external_dir, 0);
    vox::AndroidPlatformContext::android_external_storage_directory = std::string(external_dir_cstr) + "/";
    env->ReleaseStringUTFChars(external_dir, external_dir_cstr);

    const char *temp_dir_cstr = env->GetStringUTFChars(temp_dir, 0);
    vox::AndroidPlatformContext::android_temp_directory = std::string(temp_dir_cstr) + "/";
    env->ReleaseStringUTFChars(temp_dir, temp_dir_cstr);
}

JNIEXPORT void JNICALL
Java_com_khronos_vulkan_1samples_SampleLauncherActivity_sendArgumentsToPlatform(JNIEnv *env, jobject thiz, jobjectArray arg_strings) {
    std::vector<std::string> args;

    for (int i = 0; i < env->GetArrayLength(arg_strings); i++) {
        jstring arg_string = (jstring)(env->GetObjectArrayElement(arg_strings, i));

        const char *arg = env->GetStringUTFChars(arg_string, 0);

        args.push_back(std::string(arg));

        env->ReleaseStringUTFChars(arg_string, arg);
    }

    vox::AndroidPlatformContext::android_arguments = args;
}
}

namespace vox {
std::string AndroidPlatformContext::android_external_storage_directory = {};
std::string AndroidPlatformContext::android_temp_directory = {};
std::vector<std::string> AndroidPlatformContext::android_arguments = {};

AndroidPlatformContext::AndroidPlatformContext(android_app *app) : PlatformContext{}, app{app} {
    _external_storage_directory = android_external_storage_directory;
    _temp_directory = android_temp_directory;
    _arguments = android_arguments;
}
}// namespace vox