from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 加载通用抠图模型
universal_matting = pipeline(Tasks.universal_matting, model='damo/cv_unet_universal-matting')

# 加载人像抠图模型
portrait_matting = pipeline(Tasks.portrait_matting, model='damo/cv_unet_image-matting')

# 加载AI绘画 卡通模型
img_cartoon = pipeline(Tasks.image_portrait_stylization, model='damo/cv_unet_person-image-cartoon_compound-models')

# 加载AI绘画 卡通模型-素描
img_sketch_cartoon = pipeline(Tasks.image_portrait_stylization, model='damo/cv_unet_person-image-cartoon-sketch_compound-models')

# 加载AI绘画 卡通模型-3d
img_3d_cartoon = pipeline(Tasks.image_portrait_stylization,model='damo/cv_unet_person-image-cartoon-3d_compound-models')

# 加载AI绘画 卡通模型-艺术
img_artstyle_cartoon = pipeline(Tasks.image_portrait_stylization, model='damo/cv_unet_person-image-cartoon-artstyle_compound-models')

# 加载AI绘画 卡通模型-手绘
img_handdrawn_cartoon = pipeline(Tasks.image_portrait_stylization, model='damo/cv_unet_person-image-cartoon-handdrawn_compound-models')

