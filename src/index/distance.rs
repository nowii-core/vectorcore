#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
}

impl DistanceMetric {
    pub fn to_u8(self) -> u8 {
        match self {
            DistanceMetric::Cosine => 1,
            DistanceMetric::Euclidean => 2,
            DistanceMetric::DotProduct => 3,
        }
    }

    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(DistanceMetric::Cosine),
            2 => Some(DistanceMetric::Euclidean),
            3 => Some(DistanceMetric::DotProduct),
            _ => None,
        }
    }
}

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimension mismatch: {} vs {}", a.len(), b.len());
    let len = a.len();
    let chunks = len / 4;
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;

    for i in 0..chunks {
        let idx = i * 4;
        s0 += a[idx] * b[idx];
        s1 += a[idx + 1] * b[idx + 1];
        s2 += a[idx + 2] * b[idx + 2];
        s3 += a[idx + 3] * b[idx + 3];
    }

    let mut tail = 0.0f32;
    for i in (chunks * 4)..len {
        tail += a[i] * b[i];
    }

    (s0 + s1) + (s2 + s3) + tail
}

pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimension mismatch: {} vs {}", a.len(), b.len());
    let len = a.len();
    let chunks = len / 4;
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;

    for i in 0..chunks {
        let idx = i * 4;
        let d0 = a[idx] - b[idx];
        let d1 = a[idx + 1] - b[idx + 1];
        let d2 = a[idx + 2] - b[idx + 2];
        let d3 = a[idx + 3] - b[idx + 3];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }

    let mut tail = 0.0f32;
    for i in (chunks * 4)..len {
        let d = a[i] - b[i];
        tail += d * d;
    }

    (s0 + s1) + (s2 + s3) + tail
}

pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    l2_squared(a, b).sqrt()
}

pub fn norm(a: &[f32]) -> f32 {
    dot_product(a, a).sqrt()
}

pub fn normalize_in_place(v: &mut [f32]) {
    let n = norm(v);
    if n > f32::EPSILON {
        let inv = 1.0 / n;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let denom = norm(a) * norm(b);
    if denom <= f32::EPSILON {
        return 0.0;
    }
    (dot_product(a, b) / denom).clamp(-1.0, 1.0)
}

pub fn similarity(metric: DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        DistanceMetric::Cosine => cosine_similarity(a, b),
        DistanceMetric::DotProduct => dot_product(a, b),
        DistanceMetric::Euclidean => 1.0 / (1.0 + l2_distance(a, b)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_metrics() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
        assert!((l2_distance(&a, &b) - 0.0).abs() < 1e-6);

        let ortho_a = vec![1.0, 0.0, 0.0, 0.0];
        let ortho_b = vec![0.0, 1.0, 0.0, 0.0];
        assert!(cosine_similarity(&ortho_a, &ortho_b).abs() < 1e-6);
        assert!((l2_distance(&ortho_a, &ortho_b) - std::f32::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_in_place() {
        let mut v = vec![3.0, 4.0];
        normalize_in_place(&mut v);
        assert!((norm(&v) - 1.0).abs() < 1e-6);
    }
}
