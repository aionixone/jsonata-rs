use glob::glob;
use std::env;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::collections::HashSet;

fn main() {
    println!("cargo:rerun-if-changed=tests/testsuite/**/*.json");
    println!("cargo:rerun-if-changed=tests/customsuite/**/*.json");

    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("generated_tests.rs");
    let mut file = fs::File::create(dest_path).unwrap();

    let resources = {
        let mut r = get_test_resources("tests/testsuite/**/*.json");
        r.extend(get_test_resources("tests/customsuite/**/*.json"));
        r
    };

    // Optional: Enable specific groups under `skip/` by setting CSV env `JSONATA_ENABLE_GROUPS`.
    // Examples:
    //   JSONATA_ENABLE_GROUPS=function-typeOf,function-average
    //   JSONATA_ENABLE_GROUPS=all
    let enabled_groups_env = env::var("JSONATA_ENABLE_GROUPS").unwrap_or_default();
    let enable_all = enabled_groups_env.trim().eq_ignore_ascii_case("all");
    let enabled_groups: HashSet<String> = enabled_groups_env
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    for resource in resources {
        let is_skip = resource.contains("/skip/");
        let should_ignore = if is_skip {
            if enable_all {
                false
            } else if enabled_groups.is_empty() {
                true
            } else {
                let group = extract_group_from_skip_path(&resource);
                !enabled_groups.contains(&group)
            }
        } else {
            false
        };

        if should_ignore {
            writeln!(
                file,
                r#"
                #[test]
                #[ignore]
                fn test_{}() {{
                    test_case(r"{}");
                }}
                "#,
                sanitize_filename(&resource),
                resource
            )
            .unwrap();
        } else {
            writeln!(
                file,
                r#"
                #[test]
                fn test_{}() {{
                    test_case(r"{}");
                }}
                "#,
                sanitize_filename(&resource),
                resource
            )
            .unwrap();
        }
    }
}

fn get_test_resources(pattern: &str) -> Vec<String> {
    glob(pattern)
        .expect("Failed to read glob pattern")
        .filter_map(Result::ok)
        .filter(|path| !path.to_string_lossy().contains("datasets")) // Exclude datasets folder
        .map(|path| path.to_string_lossy().into_owned())
        .collect()
}

fn sanitize_filename(filename: &str) -> String {
    let mut sanitized = String::new();
    let mut prev_was_underscore = false;

    for c in filename.chars() {
        if c.is_alphanumeric() {
            if prev_was_underscore {
                sanitized.push('_');
                prev_was_underscore = false;
            }
            sanitized.push(c.to_ascii_lowercase());
        } else {
            prev_was_underscore = true;
        }
    }

    sanitized
}

/// Extract the logical group name from a `.../skip/...` resource path.
///
/// Patterns handled:
/// - tests/testsuite/skip/<group>/caseXXX.json -> group = segment after `skip`
/// - tests/customsuite/<group>/skip/caseXXX.json -> group = segment before `skip`
fn extract_group_from_skip_path(resource: &str) -> String {
    let parts: Vec<&str> = resource.split('/').collect();
    if let Some((idx, _)) = parts.iter().enumerate().find(|(_, s)| **s == "skip") {
        // testsuite style: skip/<group>/...
        if idx + 2 < parts.len() {
            return parts[idx + 1].to_string();
        }
        // customsuite style: .../<group>/skip/<file>
        if idx >= 1 {
            return parts[idx - 1].to_string();
        }
    }
    // Fallback: whole path as group, unlikely but safe
    resource.to_string()
}
