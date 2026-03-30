# tests/test_generate_teacher_data.py
import unittest


class TestGenerateTeacherDataCli(unittest.TestCase):
    def test_defaults(self):
        import generate_teacher_data

        args = generate_teacher_data.parse_args([])
        self.assertEqual(args.teacher_model_id, "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        self.assertEqual(args.dataset, "qwedsacf/competition_math")
        self.assertEqual(args.dataset_split, "train[:7500]")
        self.assertEqual(args.output_dir, "/data/yequan/fura/kd/")
        self.assertEqual(args.max_tokens, 1024)
        self.assertEqual(args.top_k, 256)
        self.assertEqual(args.prompt_template, "boxed.prompt")
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.chunk_size, 500)

    def test_custom_args(self):
        import generate_teacher_data

        args = generate_teacher_data.parse_args([
            "--teacher-model-id", "some/model",
            "--top-k", "64",
            "--max-tokens", "512",
            "--chunk-size", "100",
        ])
        self.assertEqual(args.teacher_model_id, "some/model")
        self.assertEqual(args.top_k, 64)
        self.assertEqual(args.max_tokens, 512)
        self.assertEqual(args.chunk_size, 100)

    def test_resolve_output_path(self):
        import generate_teacher_data

        args = generate_teacher_data.parse_args([
            "--teacher-model-id", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "--dataset", "qwedsacf/competition_math",
        ])
        path = generate_teacher_data.resolve_output_path(args)
        self.assertEqual(
            path,
            "/data/yequan/fura/kd/DeepSeek-R1-Distill-Qwen-7B-competition_math",
        )

    def test_resolve_output_path_strips_org(self):
        import generate_teacher_data

        args = generate_teacher_data.parse_args([
            "--teacher-model-id", "org/ModelName",
            "--dataset", "user/dataset_name",
        ])
        path = generate_teacher_data.resolve_output_path(args)
        self.assertEqual(path, "/data/yequan/fura/kd/ModelName-dataset_name")


if __name__ == "__main__":
    unittest.main()
