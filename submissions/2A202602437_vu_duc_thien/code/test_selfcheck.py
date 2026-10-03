"""test_selfcheck.py - kiểm tra tự viết cho các phần dễ sai (RUBRIC mục H). Chạy trên CPU, không train.

    cd submissions/<mssv>_<ten>/code && python -m unittest test_selfcheck -v

Các test dùng model timm với pretrained=False (không cần tải trọng số). Test đọc ảnh thật chỉ chạy khi
biến môi trường DEEPWEEDS_DIR trỏ tới thư mục chứa images/ và labels/.
"""
import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as B  # noqa: E402
import dataset as D  # noqa: E402
import inference as I  # noqa: E402
import losses as L  # noqa: E402
import model as M  # noqa: E402
import train as TR  # noqa: E402
from eval import compute_metrics, parse_seed  # noqa: E402  (train.py đã thêm gốc repo vào sys.path)


def _randomize_bn(model, seed=0):
    """BN mới tạo có mean 0 / var 1 nên gộp BN gần như không đổi gì; đặt thống kê ngẫu nhiên để test có nghĩa."""
    g = torch.Generator().manual_seed(seed)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.copy_(torch.randn(m.num_features, generator=g) * 0.1)
            m.running_var.copy_(torch.rand(m.num_features, generator=g) + 0.5)
            m.weight.data.copy_(torch.rand(m.num_features, generator=g) + 0.5)
            m.bias.data.copy_(torch.randn(m.num_features, generator=g) * 0.1)


class TestLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(64, 9) * 3
        self.y = torch.randint(0, 9, (64,))

    def test_focal_gamma0_equals_ce(self):
        fl = L.FocalLoss(gamma=0.0)(self.logits, self.y)
        ce = F.cross_entropy(self.logits, self.y)
        self.assertLess(abs(fl.item() - ce.item()), 1e-6)

    def test_focal_downweights_easy_examples(self):
        self.assertLess(L.FocalLoss(gamma=2.0)(self.logits, self.y).item(),
                        F.cross_entropy(self.logits, self.y).item())

    def test_focal_alpha_is_class_weight(self):
        alpha = torch.rand(9) + 0.5
        fl = L.FocalLoss(gamma=0.0, alpha=alpha)(self.logits, self.y)
        manual = (F.cross_entropy(self.logits, self.y, reduction="none") * alpha[self.y]).mean()
        self.assertLess(abs(fl.item() - manual.item()), 1e-6)

    def test_label_smoothing(self):
        self.assertLess(abs(L.LabelSmoothingCE(0.0)(self.logits, self.y).item()
                            - F.cross_entropy(self.logits, self.y).item()), 1e-6)
        self.assertLess(abs(L.LabelSmoothingCE(0.1)(self.logits, self.y).item()
                            - F.cross_entropy(self.logits, self.y, label_smoothing=0.1).item()), 1e-6)

    def test_class_weights(self):
        counts = np.array([1000, 500, 250, 100, 100, 100, 100, 100, 5000])
        w = L.class_weights(counts, 0.0).numpy()
        self.assertAlmostEqual(w.mean(), 1.0, places=5)
        np.testing.assert_allclose(w * counts, (w * counts)[0], rtol=1e-5)  # tỉ lệ nghịch số ảnh
        wb = L.class_weights(counts, 0.999).numpy()
        self.assertAlmostEqual(wb.sum(), 9.0, places=4)
        self.assertGreater(wb[3], wb[8])  # lớp hiếm nặng hơn lớp nhiều ảnh

    def test_build_criterion(self):
        self.assertIsInstance(L.build_criterion("ce"), nn.CrossEntropyLoss)
        with self.assertRaises(ValueError):
            L.build_criterion("ce_weighted")
        with self.assertRaises(ValueError):
            L.build_criterion("abc")


class TestMix(unittest.TestCase):
    def _batch(self, n=8, s=32):
        x = torch.arange(n, dtype=torch.float32).view(n, 1, 1, 1).expand(n, 3, s, s).clone()
        return x, torch.arange(n)

    def test_cutmix_lambda_is_true_box_area(self):
        np.random.seed(1)
        torch.manual_seed(1)
        for _ in range(50):
            x, y = self._batch()
            xm, (ya, yb, lam) = L.mix_batch(x, y, alpha=1.0, mode="cutmix")
            self.assertTrue(torch.equal(ya, y))
            for i in range(len(y)):
                if yb[i] == ya[i]:
                    continue  # hoán vị giữ nguyên chỗ này: không đo được diện tích
                own = (xm[i] == float(i)).float().mean().item()
                self.assertAlmostEqual(own, lam, places=6)  # lam = phần ảnh gốc còn lại, kể cả khi hộp bị cắt ở biên
                self.assertTrue(set(xm[i].unique().tolist()) <= {float(i), float(yb[i])})

    def test_mixup_mixes_images_and_labels(self):
        np.random.seed(2)
        torch.manual_seed(2)
        x, y = self._batch()
        xm, (ya, yb, lam) = L.mix_batch(x, y, alpha=0.4, mode="mixup")
        expected = lam * x + (1 - lam) * x[yb]
        self.assertTrue(torch.allclose(xm, expected))

    def test_mixed_loss_weights_two_targets(self):
        logits, ya, yb = torch.randn(4, 9), torch.tensor([0, 1, 2, 3]), torch.tensor([4, 5, 6, 7])
        ce = nn.CrossEntropyLoss()
        got = L.mixed_loss(ce, logits, (ya, yb, 0.3))
        self.assertAlmostEqual(got.item(), 0.3 * ce(logits, ya).item() + 0.7 * ce(logits, yb).item(), places=5)


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.net = M.build_model("resnet18", pretrained=False)

    def test_param_groups_no_decay_on_norm_and_bias(self):
        groups = {g["name"]: g for g in M.param_groups(self.net, 1e-4, 1e-3, 0.05)}
        self.assertEqual(groups["backbone_no_decay"]["weight_decay"], 0.0)
        self.assertTrue(all(p.ndim <= 1 for p in groups["backbone_no_decay"]["params"]))
        self.assertTrue(all(p.ndim > 1 for p in groups["backbone_decay"]["params"]))
        self.assertEqual(groups["head_decay"]["lr"], 1e-3)
        self.assertEqual(groups["head_no_decay"]["weight_decay"], 0.0)
        n_grouped = sum(p.numel() for g in groups.values() for p in g["params"])
        self.assertEqual(n_grouped, sum(p.numel() for p in self.net.parameters()))

    def test_vit_pos_embed_has_no_decay(self):
        vit = M.build_model("deit_tiny_patch16_224", pretrained=False)
        no_decay = {id(p) for g in M.param_groups(vit, 1e-4, 1e-3, 0.05) if g["weight_decay"] == 0 for p in g["params"]}
        self.assertIn(id(vit.pos_embed), no_decay)
        self.assertIn(id(vit.cls_token), no_decay)

    def test_frozen_backbone_trains_only_head_and_keeps_bn_eval(self):
        net = M.build_model("resnet18", pretrained=False, init="frozen")
        trainable = [n for n, p in net.named_parameters() if p.requires_grad]
        self.assertEqual(sorted(trainable), ["fc.bias", "fc.weight"])
        groups = M.param_groups(net, 1e-4, 1e-3, 0.05)
        self.assertTrue(all(g["name"].startswith("head") for g in groups))
        M.set_train_mode(net)
        self.assertTrue(net.training)
        self.assertTrue(all(not m.training for m in net.modules() if isinstance(m, nn.BatchNorm2d)))

    def test_param_and_gmac_count(self):
        r50 = M.build_model("resnet50", pretrained=False)
        self.assertAlmostEqual(M.count_params(r50), 23.53, delta=0.05)  # 25,6M với head 1000 lớp; head 9 lớp
        self.assertAlmostEqual(M.count_gmacs(r50, 224), 4.1, delta=0.15)  # slide: 4,1 GMAC

    def test_weight_tag_recorded(self):
        self.assertIn("scratch", M.build_model("resnet18", pretrained=True, init="scratch").weight_tag)


class TestTrainHelpers(unittest.TestCase):
    def test_config_defaults_match_guide(self):
        c = TR.Config()
        self.assertEqual((c.epochs, c.batch_size, c.lr_backbone, c.lr_head, c.weight_decay),
                         (12, 64, 1e-4, 1e-3, 0.05))
        self.assertFalse(c.save_test_predictions)
        self.assertEqual(parse_seed(str(TR.pred_path(TR.Config(exp_id="F01", seed=2), "test"))), 2)

    def test_scheduler_warmup_then_cosine(self):
        p = nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([{"params": [p], "lr": 1e-4}])
        cfg = TR.Config(epochs=3, warmup_epochs=1.0)
        sched = TR.build_scheduler(opt, cfg, steps_per_epoch=10)
        lrs = []
        for _ in range(30):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()
        self.assertAlmostEqual(lrs[0], 1e-5)          # bước 1 của warmup: 1/10 LR
        self.assertAlmostEqual(lrs[9], 1e-4)          # hết warmup: đủ LR
        self.assertTrue(all(a >= b for a, b in zip(lrs[9:], lrs[10:])))  # cosine giảm dần
        self.assertLess(lrs[-1], 1e-6 + 1e-4 * 0.01)  # về gần 0

    def test_ema_formula_and_buffers(self):
        net = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        ema = TR.EMA(net, decay=0.5)
        ema.updates = 100  # bỏ qua warmup của decay
        w0 = ema.module[0].weight.clone()
        with torch.no_grad():
            net[0].weight.add_(1.0)
            net[1].running_mean.add_(2.0)
        ema.update(net)
        self.assertTrue(torch.allclose(ema.module[0].weight, 0.5 * w0 + 0.5 * net[0].weight))
        self.assertTrue(torch.allclose(ema.module[1].running_mean, torch.full((2,), 1.0)))
        self.assertFalse(ema.module.training)

    def test_parse_overrides(self):
        d = TR.parse_overrides(["seed=3", "loss=focal", "ema_decay=0.999", "sampler=none",
                                "amp=false", "label_smoothing=0.1"])
        self.assertEqual(d, {"seed": 3, "loss": "focal", "ema_decay": 0.999, "sampler": None,
                             "amp": False, "label_smoothing": 0.1})
        with self.assertRaises(ValueError):
            TR.parse_overrides(["khong_co=1"])
        with self.assertRaises(ValueError):
            TR.parse_overrides(["epochs=none"])

    def test_metrics_match_eval(self):
        rng = np.random.default_rng(0)
        logits, y = rng.normal(size=(100, 9)), rng.integers(0, 9, 100)
        m = TR.metrics_from_logits(y, logits)
        p = TR.softmax(logits)
        self.assertEqual(m["macro_f1"], compute_metrics(y, p.argmax(1), p)["macro_f1"])


class TestInference(unittest.TestCase):
    def test_fuse_conv_bn_exact(self):
        for name in ("resnet18", "efficientnet_b0", "mobilenetv3_large_100"):
            with self.subTest(name=name):
                torch.manual_seed(0)
                net = M.build_model(name, pretrained=False).eval()
                _randomize_bn(net)
                fused = I.fuse_conv_bn(net)
                self.assertGreater(fused.fused_pairs, 10)
                self.assertFalse(any(isinstance(m, nn.BatchNorm2d) for m in fused.modules()))
                self.assertLess(I.check_fusion(net, fused, 64), 1e-4)

    def test_fuse_skips_models_without_bn(self):
        net = M.build_model("convnext_atto", pretrained=False)
        self.assertEqual(I.fuse_conv_bn(net).fused_pairs, 0)

    def test_temperature_recovers_scale(self):
        rng = np.random.default_rng(0)
        n = 5000
        y = rng.integers(0, 9, n)
        true_logits = rng.normal(size=(n, 9))
        true_logits[np.arange(n), y] += 2.0
        # Nhãn lấy mẫu từ softmax(true_logits): mô hình "true" đã hiệu chuẩn; nhân logit với 3 -> quá tự tin
        p = I._softmax(true_logits)
        y = np.array([rng.choice(9, p=row) for row in p])
        T = I.fit_temperature(true_logits * 3.0, y)
        self.assertAlmostEqual(T, 3.0, delta=0.25)
        probs = I.apply_temperature(true_logits * 3.0, T)
        np.testing.assert_array_equal(probs.argmax(1), true_logits.argmax(1))  # accuracy không đổi

    def test_aggregate_views(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=(10, 9)), rng.normal(size=(10, 9))
        np.testing.assert_allclose(I.aggregate_views([a], "prob"), I._softmax(a))
        np.testing.assert_allclose(I.aggregate_views([a, b], "prob"), (I._softmax(a) + I._softmax(b)) / 2)
        np.testing.assert_allclose(I.aggregate_views([a, b], "logit"), I._softmax((a + b) / 2))
        np.testing.assert_allclose(I.ensemble_probs([I._softmax(a), I._softmax(b)]).sum(1), 1.0)

    def test_views(self):
        x = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).view(2, 3, 8, 8)
        crops = I.views_multicrop(x, 6, flip=True)
        self.assertEqual(len(crops), 10)
        self.assertTrue(all(c.shape == (2, 3, 6, 6) for c in crops))
        self.assertTrue(torch.equal(crops[4], x[..., 1:7, 1:7]))
        self.assertTrue(torch.equal(I.view_hflip(I.view_hflip(x)), x))
        self.assertEqual([v.shape[-1] for v in I.views_multiscale(x, [4, 8, 12])], [4, 8, 12])

    def test_predict_logits_keeps_order(self):
        net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 9)).eval()
        xs = torch.randn(10, 3, 4, 4)
        loader = [(xs[i:i + 4], torch.arange(i, min(i + 4, 10)), [f"f{j}" for j in range(i, min(i + 4, 10))])
                  for i in range(0, 10, 4)]
        names, y, logits = I.predict_logits(net, loader, "cpu", channels_last=False)
        self.assertEqual(names, [f"f{j}" for j in range(10)])
        with torch.no_grad():
            np.testing.assert_allclose(logits, net(xs).numpy(), atol=1e-6)
        _, _, two = I.predict_logits(net, loader, "cpu", view=I.views_hflip_pair, channels_last=False)
        self.assertEqual(len(two), 2)


class TestBenchmark(unittest.TestCase):
    def test_bench_percentiles_and_sync(self):
        calls = []
        r = B.bench(lambda: calls.append(1), warmup=10, iters=50, sync=lambda: None)
        self.assertEqual(len(calls), 60)  # warmup 10 + đo 50
        self.assertLessEqual(r["p50"], r["p95"])
        self.assertLessEqual(r["p95"], r["p99"])

    def test_latency_report_cpu(self):
        net = nn.Sequential(nn.Conv2d(3, 4, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 9))
        r = B.latency_report(net, 1, 32, "fp32", "cpu", warmup=2, iters=5)
        self.assertEqual((r["batch"], r["dtype"], r["gpu"]), (1, "fp32", "cpu"))


DATA = os.environ.get("DEEPWEEDS_DIR")


@unittest.skipUnless(DATA, "đặt DEEPWEEDS_DIR=<thư mục có images/ và labels/> để chạy")
class TestDataset(unittest.TestCase):
    def test_split_checks_and_loader(self):
        tr, va, te = D.load_split(f"{DATA}/labels")
        info = D.check_split(tr, va, te, f"{DATA}/images", verbose=False)
        self.assertEqual(info["union"], D.TOTAL_IMAGES)
        self.assertEqual(sum(info["overlap"].values()), 0)
        tf = D.build_transforms(False, 224)
        ds = D.DeepWeedsDataset(va.iloc[:4], f"{DATA}/images", tf)
        img, label, name = ds[0]
        self.assertEqual(tuple(img.shape), (3, 224, 224))
        self.assertEqual((label, name), (int(va.iloc[0]["Label"]), va.iloc[0]["Filename"]))
        self.assertTrue(torch.equal(ds[1][0], ds[1][0]))  # eval: không ngẫu nhiên
        loader = D.make_loader(va.iloc[:10], f"{DATA}/images", tf, 4, train=False, num_workers=0)
        self.assertEqual([n for _, _, f in loader for n in f], list(va.iloc[:10]["Filename"]))
        for aug in D.AUGS:
            x, _, _ = D.DeepWeedsDataset(tr.iloc[:1], f"{DATA}/images", D.build_transforms(True, 224, aug))[0]
            self.assertEqual(tuple(x.shape), (3, 224, 224))
        bal = D.make_loader(tr, f"{DATA}/images", tf, 64, train=True, sampler="balanced", num_workers=0)
        idx = np.array(list(iter(bal.sampler)))
        share = np.bincount(tr["Label"].to_numpy()[idx], minlength=9) / len(idx)
        self.assertTrue(np.all(np.abs(share - 1 / 9) < 0.02))  # sampler cân bằng: mỗi lớp ~1/9


if __name__ == "__main__":
    unittest.main()
