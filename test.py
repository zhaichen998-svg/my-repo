"""
Test script to validate FSS Network implementation
"""

import torch
import sys

def test_feature_extractor():
    """Test Feature Extraction Module"""
    print("Testing Feature Extraction Module...")
    from models.modules import FeatureExtractor
    
    model = FeatureExtractor(pretrained=False)
    x = torch.randn(2, 3, 400, 400)
    features = model(x)
    
    assert 'low' in features, "Missing low-level features"
    assert 'mid' in features, "Missing mid-level features"
    assert 'high' in features, "Missing high-level features"
    assert features['low'].shape[1] == 256, f"Expected 256 channels, got {features['low'].shape[1]}"
    assert features['mid'].shape[1] == 1024, f"Expected 1024 channels, got {features['mid'].shape[1]}"
    assert features['high'].shape[1] == 512, f"Expected 512 channels, got {features['high'].shape[1]}"
    
    print("✓ Feature Extraction Module passed")
    return True


def test_acm():
    """Test Attention Calibration Module"""
    print("Testing ACM Module...")
    from models.modules import ACM
    
    model = ACM(channels=256)
    x = torch.randn(2, 256, 50, 50)
    out = model(x)
    
    assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"
    assert hasattr(model, 'alpha'), "Missing learnable alpha coefficient"
    assert hasattr(model, 'beta'), "Missing learnable beta coefficient"
    
    print("✓ ACM Module passed")
    return True


def test_sfem():
    """Test SFEM Module"""
    print("Testing SFEM Module...")
    from models.modules import SFEM
    
    model = SFEM(channels=512)
    query_feat = torch.randn(2, 512, 25, 25)
    support_feat = torch.randn(2, 512, 25, 25)
    support_mask = torch.randint(0, 2, (2, 25, 25)).float()
    
    query_enh, support_enh = model(query_feat, support_feat, support_mask)
    
    assert query_enh.shape == query_feat.shape, f"Query shape mismatch"
    assert support_enh.shape == support_feat.shape, f"Support shape mismatch"
    
    print("✓ SFEM Module passed")
    return True


def test_feature_matching():
    """Test Feature Matching Module"""
    print("Testing Feature Matching Module...")
    from models.modules import FeatureMatching
    
    model = FeatureMatching(channels=512)
    query_feat = torch.randn(2, 512, 25, 25)
    support_feat = torch.randn(2, 512, 25, 25)
    support_mask = torch.randint(0, 2, (2, 25, 25)).float()
    
    matched_feat, similarity = model(query_feat, support_feat, support_mask)
    
    assert matched_feat.shape == query_feat.shape, f"Matched feature shape mismatch"
    assert similarity.shape == (2, 1, 25, 25), f"Expected similarity shape (2, 1, 25, 25), got {similarity.shape}"
    
    print("✓ Feature Matching Module passed")
    return True


def test_hpkim():
    """Test HPKIM Module"""
    print("Testing HPKIM Module...")
    from models.modules import HPKIM
    
    model = HPKIM(low_channels=256, mid_channels=1024, high_channels=512)
    
    query_feats = {
        'low': torch.randn(2, 256, 100, 100),
        'mid': torch.randn(2, 1024, 50, 50),
        'high': torch.randn(2, 512, 25, 25)
    }
    
    support_feats = {
        'low': torch.randn(2, 256, 100, 100),
        'mid': torch.randn(2, 1024, 50, 50),
        'high': torch.randn(2, 512, 25, 25)
    }
    
    output = model(query_feats, support_feats, target_size=(400, 400))
    
    assert output.shape == (2, 256, 400, 400), f"Expected shape (2, 256, 400, 400), got {output.shape}"
    
    print("✓ HPKIM Module passed")
    return True


def test_fss_network():
    """Test complete FSS Network"""
    print("Testing FSS Network...")
    from models import FSSNetwork
    
    model = FSSNetwork(pretrained=False)
    
    query_img = torch.randn(2, 3, 400, 400)
    support_img = torch.randn(2, 3, 400, 400)
    support_mask = torch.randint(0, 2, (2, 400, 400)).float()
    
    pred = model(query_img, support_img, support_mask)
    
    assert pred.shape == (2, 2, 400, 400), f"Expected shape (2, 2, 400, 400), got {pred.shape}"
    
    # Test prediction method
    pred_mask = model.predict(query_img, support_img, support_mask)
    assert pred_mask.shape == (2, 400, 400), f"Expected mask shape (2, 400, 400), got {pred_mask.shape}"
    
    print("✓ FSS Network passed")
    return True


def test_dataset():
    """Test Dataset class"""
    print("Testing Dataset...")
    from data import SegPPDDataset, MetaLearningDataLoader
    import os
    import tempfile
    from PIL import Image
    import numpy as np
    
    # Create temporary dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = os.path.join(tmpdir, 'images')
        mask_dir = os.path.join(tmpdir, 'masks')
        os.makedirs(img_dir)
        os.makedirs(mask_dir)
        
        # Create dummy images
        for i in range(5):
            img = Image.fromarray(np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8))
            img.save(os.path.join(img_dir, f'img{i}.jpg'))
            
            mask = Image.fromarray(np.random.randint(0, 2, (400, 400), dtype=np.uint8) * 255)
            mask.save(os.path.join(mask_dir, f'img{i}.png'))
        
        # Test dataset
        dataset = SegPPDDataset(root_dir=tmpdir, split='train', img_size=400)
        assert len(dataset) == 5, f"Expected 5 samples, got {len(dataset)}"
        
        sample = dataset[0]
        assert 'image' in sample, "Missing image key"
        assert 'mask' in sample, "Missing mask key"
        assert sample['image'].shape == (3, 400, 400), f"Wrong image shape: {sample['image'].shape}"
        assert sample['mask'].shape == (400, 400), f"Wrong mask shape: {sample['mask'].shape}"
        
        # Test meta-learning dataloader
        meta_loader = MetaLearningDataLoader(dataset, batch_size=2, n_way=1, k_shot=1, num_workers=0)
        episode = meta_loader.create_episode()
        
        assert 'query_img' in episode, "Missing query_img"
        assert 'support_img' in episode, "Missing support_img"
        assert 'support_mask' in episode, "Missing support_mask"
    
    print("✓ Dataset passed")
    return True


def test_metrics():
    """Test metrics"""
    print("Testing Metrics...")
    from utils import compute_iou, compute_dice
    
    # Perfect prediction
    pred = torch.zeros(2, 2, 100, 100)
    pred[:, 1, 50:, 50:] = 1  # Class 1 in bottom-right quadrant
    target = torch.zeros(2, 100, 100)
    target[:, 50:, 50:] = 1
    
    iou = compute_iou(pred, target)
    dice = compute_dice(pred, target)
    
    assert iou > 0.4, f"IoU too low: {iou}"
    assert dice > 0.5, f"Dice too low: {dice}"
    
    print("✓ Metrics passed")
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("FSS Network Validation Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_feature_extractor,
        test_acm,
        test_sfem,
        test_feature_matching,
        test_hpkim,
        test_fss_network,
        test_dataset,
        test_metrics
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {str(e)}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
