//  Copyright (c) 2022 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "base/singleton.h"
#include "math/spherical_harmonics3.h"
#include "framework/rendering/postprocessing_computepass.h"
#include "framework/rendering/postprocessing_pipeline.h"
#include "framework/shader/shader_data.h"
#include "texture/texture.h"

namespace vox {
class TextureManager : public Singleton<TextureManager> {
public:
    static TextureManager &GetSingleton();

    static TextureManager *GetSingletonPtr();

    explicit TextureManager(core::Device &device);

    ~TextureManager() = default;

    void CollectGarbage();

public:
    /**
     * @brief Loads in a ktx 2D texture
     */
    std::shared_ptr<Texture> LoadTexture(const std::string &file);

    /**
     * @brief Loads in a ktx 2D texture array
     */
    std::shared_ptr<Texture> LoadTextureArray(const std::string &file);

    /**
     * @brief Loads in a ktx 2D texture cubemap
     */
    std::shared_ptr<Texture> LoadTextureCubemap(const std::string &file);

    void UploadTexture(Texture *image);

public:
    std::shared_ptr<Texture> GenerateIBL(const std::string &file, rendering::RenderContext &render_context);

    SphericalHarmonics3 GenerateSH(const std::string &file);

private:
    core::Device &device_;
    std::unordered_map<std::string, std::shared_ptr<Texture>> image_pool_;
    VkSamplerCreateInfo sampler_create_info_;
    std::unique_ptr<core::Sampler> sampler_{nullptr};

    ShaderData shader_data_;
    std::unique_ptr<rendering::PostProcessingPipeline> pipeline_{nullptr};
    rendering::PostProcessingComputePass *ibl_pass_{nullptr};
};

template<>
inline TextureManager *Singleton<TextureManager>::ms_singleton{nullptr};

}// namespace vox
