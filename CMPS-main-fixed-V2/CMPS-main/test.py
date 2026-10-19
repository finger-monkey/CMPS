import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor


MODE = 'bilinear'


def _forward_features(net, inputs, modal=1):
    if hasattr(net, 'visible_module') and hasattr(net, 'thermal_module'):
        output = net(inputs, modal=modal)
    else:
        output = net(inputs)
    return output[0] if isinstance(output, (tuple, list)) else output


def _forward_batch(net, inputs, camera_ids, dataset, fixed_modal=None):
    if fixed_modal is not None or not (hasattr(net, 'visible_module') and hasattr(net, 'thermal_module')):
        return _forward_features(net, inputs, modal=fixed_modal or 1)
    visible_cameras = set(getattr(dataset, 'visible_cameras', ()))
    infrared_cameras = set(getattr(dataset, 'infrared_cameras', ()))
    camera_ids = camera_ids.tolist() if torch.is_tensor(camera_ids) else list(camera_ids)
    visible_indices = [index for index, camera_id in enumerate(camera_ids) if camera_id in visible_cameras]
    infrared_indices = [index for index, camera_id in enumerate(camera_ids) if camera_id in infrared_cameras]
    outputs = [None] * len(camera_ids)
    if visible_indices:
        indices = torch.as_tensor(visible_indices, device=inputs.device, dtype=torch.long)
        visible_features = _forward_features(net, inputs.index_select(0, indices), modal=1)
        for index, feature in zip(visible_indices, visible_features):
            outputs[index] = feature
    if infrared_indices:
        indices = torch.as_tensor(infrared_indices, device=inputs.device, dtype=torch.long)
        infrared_features = _forward_features(net, inputs.index_select(0, indices), modal=2)
        for index, feature in zip(infrared_indices, infrared_features):
            outputs[index] = feature
    if any(output is None for output in outputs):
        raise ValueError('Found a camera ID without a visible/infrared modality mapping.')
    return torch.stack(outputs, dim=0)


def _modality_split(items, dataset):
    visible_cameras = set(getattr(dataset, 'visible_cameras', ()))
    infrared_cameras = set(getattr(dataset, 'infrared_cameras', ()))
    mapped_cameras = visible_cameras | infrared_cameras
    unmapped_cameras = sorted({item[2] for item in items if item[2] not in mapped_cameras})
    if unmapped_cameras:
        raise ValueError('Dataset contains unmapped query/gallery camera IDs: {}'.format(unmapped_cameras))
    visible = [item for item in items if item[2] in visible_cameras]
    infrared = [item for item in items if item[2] in infrared_cameras]
    return visible, infrared


def _get_eval_sets(dataset, direction):
    query = list(dataset.query)
    gallery = list(dataset.gallery)
    if direction == 'auto':
        return [('auto', query, gallery)]

    query_visible, query_infrared = _modality_split(query, dataset)
    gallery_visible, gallery_infrared = _modality_split(gallery, dataset)

    if direction == 'both':
        if query_visible and gallery_infrared and query_infrared and gallery_visible:
            return [
                ('vis2ir', query_visible, gallery_infrared),
                ('ir2vis', query_infrared, gallery_visible),
            ]
        query_modalities = set()
        gallery_modalities = set()
        if query_visible:
            query_modalities.add('vis')
        if query_infrared:
            query_modalities.add('ir')
        if gallery_visible:
            gallery_modalities.add('vis')
        if gallery_infrared:
            gallery_modalities.add('ir')
        if query_modalities == {'vis'} and gallery_modalities == {'ir'}:
            return [('vis2ir', query, gallery), ('ir2vis', gallery, query)]
        if query_modalities == {'ir'} and gallery_modalities == {'vis'}:
            return [('vis2ir', gallery, query), ('ir2vis', query, gallery)]
        raise ValueError('Could not derive both cross-modality evaluation directions from the dataset query/gallery split.')

    if direction == 'vis2ir':
        if query_visible and gallery_infrared:
            return [('vis2ir', query_visible, gallery_infrared)]
        if gallery_visible and query_infrared:
            return [('vis2ir', gallery_visible, query_infrared)]
    elif direction == 'ir2vis':
        if query_infrared and gallery_visible:
            return [('ir2vis', query_infrared, gallery_visible)]
        if gallery_infrared and query_visible:
            return [('ir2vis', gallery_infrared, query_visible)]

    raise ValueError(
        'Could not derive {} evaluation sets. Check the dataset camera-to-modality mapping and query/gallery metadata.'.format(direction)
    )


def test(dataset, net, noise, args, evaluator, epoch, direction='auto'):
    print('>> Evaluating network on test datasets...')
    net = net.cuda()
    net.eval()
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    def add_noise(img):
        perturbation = noise.detach().cpu()
        image = img.cpu()
        perturbation = F.interpolate(
            perturbation.unsqueeze(0), mode=MODE, size=tuple(image.shape[-2:]), align_corners=True
        ).squeeze(0)
        return torch.clamp(image + perturbation, 0, 1)

    query_transform = T.Compose([
        T.RectScale(args.height, args.width),
        T.ToTensor(),
        T.Lambda(lambda img: add_noise(img)),
        normalize,
    ])
    gallery_transform = T.Compose([
        T.RectScale(args.height, args.width),
        T.ToTensor(),
        normalize,
    ])

    results = []
    for direction_name, eval_query, eval_gallery in _get_eval_sets(dataset, direction):
        if not eval_query or not eval_gallery:
            raise ValueError('The {} evaluation split has an empty query or gallery.'.format(direction_name))
        print('>> Direction: {} | Query: {} | Gallery: {}'.format(
            direction_name, len(eval_query), len(eval_gallery)
        ))
        query_loader = DataLoader(
            Preprocessor(eval_query, root=dataset.images_dir, transform=query_transform),
            batch_size=args.batch_size, num_workers=0, shuffle=False, pin_memory=True
        )
        gallery_loader = DataLoader(
            Preprocessor(eval_gallery, root=dataset.images_dir, transform=gallery_transform),
            batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True
        )
        query_features = []
        gallery_features = []
        query_modal = 1 if direction_name == 'vis2ir' else 2 if direction_name == 'ir2vis' else None
        gallery_modal = 2 if direction_name == 'vis2ir' else 1 if direction_name == 'ir2vis' else None
        with torch.no_grad():
            for inputs, _, _, camera_ids in query_loader:
                inputs = inputs.cuda(non_blocking=True)
                query_features.append(_forward_batch(net, inputs, camera_ids, dataset, fixed_modal=query_modal))
            for inputs, _, _, camera_ids in gallery_loader:
                inputs = inputs.cuda(non_blocking=True)
                gallery_features.append(_forward_batch(net, inputs, camera_ids, dataset, fixed_modal=gallery_modal))
        query_features = torch.cat(query_features, dim=0)
        gallery_features = torch.cat(gallery_features, dim=0)
        distance_matrix = calDist(query_features, gallery_features)
        result = evaluator.evaMat(distance_matrix, eval_query, eval_gallery)
        results.append((direction_name, result))
    return results


def calDist(qFeat, gFeat):
    m, n = qFeat.size(0), gFeat.size(0)
    x = qFeat.view(m, -1)
    y = gFeat.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(x, y.t(), beta=1, alpha=-2)
    return dist_m
