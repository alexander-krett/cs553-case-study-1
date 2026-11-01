from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class StaticProjectTests(unittest.TestCase):
    def test_expected_services_exist(self) -> None:
        compose = (ROOT / "docker-compose.yml").read_text()
        for service in ("api_product", "local_product", "prometheus", "grafana", "gateway"):
            self.assertRegex(compose, rf"(?m)^  {service}:$")

    def test_prometheus_scrapes_both_ports_for_both_products(self) -> None:
        config = (ROOT / "prometheus" / "prometheus.yml").read_text()
        for target in (
            "api_product:8000",
            "api_product:9100",
            "local_product:8000",
            "local_product:9100",
        ):
            self.assertIn(target, config)

    def test_application_declares_eight_metric_families(self) -> None:
        app = (ROOT / "app.py").read_text()
        metrics = set(re.findall(r'"(resumelens_[a-z_]+)"', app))
        self.assertEqual(
            metrics,
            {
                "resumelens_requests_total",
                "resumelens_request_duration_seconds",
                "resumelens_in_progress_requests",
                "resumelens_input_characters",
                "resumelens_output_characters",
                "resumelens_model_calls_total",
                "resumelens_errors_total",
                "resumelens_build_info",
            },
        )

    def test_dashboard_is_valid_and_has_eight_panels(self) -> None:
        dashboard = json.loads((ROOT / "grafana" / "dashboards" / "resumelens-overview.json").read_text())
        self.assertEqual(dashboard["uid"], "resumelens-cs3")
        self.assertEqual(len(dashboard["panels"]), 8)

    def test_secret_file_is_ignored(self) -> None:
        ignore = (ROOT / ".gitignore").read_text().splitlines()
        self.assertIn(".env", ignore)

    def test_published_host_ports_stay_in_vm_range(self) -> None:
        compose = (ROOT / "docker-compose.yml").read_text()
        mappings = re.findall(r'(?m)^      - "[^"]*:\$\{\w+:-(\d+)\}:(\d+)"$', compose)
        self.assertEqual(len(mappings), 10)
        for host_port, _ in mappings:
            self.assertTrue(2100 <= int(host_port) <= 2110, host_port)
        self.assertEqual(len(re.findall(r"(?m)^    ports:$", compose)), 6)

    def test_env_example_ports_stay_in_vm_range(self) -> None:
        env = (ROOT / ".env.example").read_text()
        ports = re.findall(r"(?m)^\w+_PORT=(\d+)$", env)
        self.assertEqual(len(ports), 10)
        for port in ports:
            self.assertTrue(2100 <= int(port) <= 2110, port)

    def test_ngrok_publishes_gateway_and_hf_token_is_not_configured(self) -> None:
        compose = (ROOT / "docker-compose.yml").read_text()
        self.assertRegex(compose, r"(?m)^  ngrok:$")
        self.assertIn("ngrok http gateway:8080", compose)
        self.assertNotIn("HF_TOKEN", compose)
        self.assertNotIn("HF_TOKEN", (ROOT / ".env.example").read_text())

    def test_gateway_has_all_required_public_paths(self) -> None:
        config = (ROOT / "gateway" / "nginx.conf").read_text()
        for path in ("/api/", "/local/", "/grafana/"):
            self.assertIn(path, config)


if __name__ == "__main__":
    unittest.main()
