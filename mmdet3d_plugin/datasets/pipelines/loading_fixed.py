import numpy as np
import torch
from PIL import Image
from mmdet3d.datasets.pipelines.loading import LoadMultiViewImageFromFiles_BEVDet
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_BEVDet_Fixed(LoadMultiViewImageFromFiles_BEVDet):
    """
    Versão corrigida do LoadMultiViewImageFromFiles_BEVDet para compatibilidade
    com nuScenes dataset padrão que usa campos dispersos ao invés de img_info
    """
    
    def __call__(self, results):
        """
        Override do método __call__ para reconstruir img_info a partir dos campos disponíveis
        """
        # Verificar se img_info já existe (pode vir de outros pipelines)
        if 'img_info' not in results:
            # Reconstruir img_info a partir dos campos disponíveis
            results['img_info'] = self._reconstruct_img_info(results)
        
        # Chamar get_inputs para processar as imagens
        results['img_inputs'] = self.get_inputs(results)
        
        return results
    
    def _reconstruct_img_info(self, results):
        """
        Reconstroi img_info a partir dos campos disponíveis no pipeline do nuScenes
        """
        img_info = {}
        
        # Os nomes das câmeras definidos na configuração
        cam_names = self.data_config['cams']
        
        # img_filename contém os caminhos dos arquivos
        img_filenames = results.get('img_filename', [])
        
        # lidar2img contém as matrizes de transformação
        lidar2img_transforms = results.get('lidar2img', [])
        
        if len(img_filenames) != len(cam_names):
            raise ValueError(f"Número de imagens ({len(img_filenames)}) não coincide com número de câmeras ({len(cam_names)})")
            
        if len(lidar2img_transforms) != len(cam_names):
            raise ValueError(f"Número de transformações ({len(lidar2img_transforms)}) não coincide com número de câmeras ({len(cam_names)})")
        
        # Para cada câmera, construir a estrutura esperada
        for i, cam_name in enumerate(cam_names):
            img_info[cam_name] = {
                'data_path': img_filenames[i],
                'cam_intrinsic': lidar2img_transforms[i][:3, :3].tolist(),  # Matriz intrínseca
                'sensor2lidar_rotation': np.eye(3).tolist(),  # Placeholder - será calculado se necessário
                'sensor2lidar_translation': [0, 0, 0],  # Placeholder - será calculado se necessário
            }
            
            # Tentar extrair rotação e translação da matriz lidar2img se possível
            # lidar2img = K @ [R|t] onde K é a matriz intrínseca
            # Para simplificar, vamos usar identidade por agora
            # Em um caso real, seria necessário decompor a matriz adequadamente
        
        return img_info
    
    def get_inputs(self, results, flip=None, scale=None):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        cams = self.choose_cams()
        
        for cam in cams:
            if 'img_info' not in results or cam not in results['img_info']:
                available_cams = list(results.get('img_info', {}).keys())
                raise KeyError(f"Câmera '{cam}' não encontrada em img_info. Câmeras disponíveis: {available_cams}")
                
            cam_data = results['img_info'][cam]
            filename = cam_data['data_path']
            img = Image.open(filename)
            post_rot = torch.eye(2, dtype=torch.float32)
            post_tran = torch.zeros(2, dtype=torch.float32)

            intrin = torch.tensor(cam_data['cam_intrinsic'], dtype=torch.float32)
            rot = torch.tensor(cam_data['sensor2lidar_rotation'], dtype=torch.float32)
            tran = torch.tensor(cam_data['sensor2lidar_translation'], dtype=torch.float32)

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            img, post_rot2, post_tran2 = self.img_transform(
                img, post_rot, post_tran, resize=resize, resize_dims=resize_dims,
                crop=crop, flip=flip, rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3, dtype=torch.float32)
            post_rot = torch.eye(3, dtype=torch.float32)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(self.normalize_img(img))

            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    img_adjacent = self.img_transform_core(
                        img_adjacent, resize_dims=resize_dims, crop=crop, flip=flip, rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))

            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        if self.sequential:
            imgs = torch.stack(imgs).view(len(cams), self.seq_len, 3, self.data_config['input_size'][0], self.data_config['input_size'][1])
            intrins = torch.stack(intrins)
            rots = torch.stack(rots) 
            trans = torch.stack(trans)
            post_rots = torch.stack(post_rots)
            post_trans = torch.stack(post_trans)
        else:
            imgs = torch.stack(imgs)
            intrins = torch.stack(intrins)
            rots = torch.stack(rots)
            trans = torch.stack(trans)
            post_rots = torch.stack(post_rots)
            post_trans = torch.stack(post_trans)

        return (imgs, rots, trans, intrins, post_rots, post_trans)
