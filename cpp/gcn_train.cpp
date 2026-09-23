// Early-stopped 2-layer GCN on a pre-normalized CSR tensor.
// The Python loader owns MetaMetro CGT conversion. This binary owns the
// epoch loop, which is the long stage on the larger assembly graphs.
//
// Job directory (little-endian host):
//   meta.txt: n d e n_pos n_mark max_epochs patience seed hidden latent loss lr
//   indptr.i32 indices.i32 values.f32 features.f32 pos.i32 mark.i32
// Writes result.txt and embedding.f32.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Meta {
  int n = 0;
  int d = 0;
  int e = 0;
  int n_pos = 0;
  int n_mark = 0;
  int max_epochs = 25;
  int patience = 5;
  int seed = 0;
  int hidden = 32;
  int latent = 16;
  int loss = 0;
  float lr = 0.01f;
};

std::vector<char> read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("missing " + path);
  }
  return std::vector<char>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

template <typename T>
std::vector<T> read_vec(const std::string& path, int count) {
  auto bytes = read_file(path);
  if (static_cast<int>(bytes.size()) != count * static_cast<int>(sizeof(T))) {
    throw std::runtime_error("size mismatch " + path);
  }
  std::vector<T> out(count);
  std::copy(bytes.begin(), bytes.end(), reinterpret_cast<char*>(out.data()));
  return out;
}

void spmm(const std::vector<int>& indptr, const std::vector<int>& indices, const std::vector<float>& values,
          const std::vector<float>& x, int n, int din, int dout_ignored, std::vector<float>& y) {
  (void)dout_ignored;
  const int d = static_cast<int>(x.size() / n);
  std::fill(y.begin(), y.end(), 0.f);
  for (int i = 0; i < n; ++i) {
    for (int p = indptr[i]; p < indptr[i + 1]; ++p) {
      const int j = indices[p];
      const float a = values[p];
      for (int k = 0; k < d; ++k) {
        y[i * d + k] += a * x[j * d + k];
      }
    }
  }
  (void)din;
}

float dot_row(const std::vector<float>& z, int i, int j, int latent) {
  float sum = 0.f;
  for (int k = 0; k < latent; ++k) {
    sum += z[i * latent + k] * z[j * latent + k];
  }
  return sum;
}

void add_bilinear(std::vector<float>& grad, const std::vector<float>& z, int i, int j, float g, int latent) {
  for (int k = 0; k < latent; ++k) {
    grad[i * latent + k] += g * z[j * latent + k];
    grad[j * latent + k] += g * z[i * latent + k];
  }
}

struct Adam {
  float lr;
  int t = 0;
  std::vector<float> m0, v0, m1, v1;
  explicit Adam(float learning, int h, int d, int latent)
      : lr(learning), m0(d * h), v0(d * h), m1(h * latent), v1(h * latent) {}
  void apply(std::vector<float>& w, std::vector<float>& m, std::vector<float>& v, const std::vector<float>& g) {
    const float b1 = 0.9f;
    const float b2 = 0.999f;
    const float bc1 = 1.f - std::pow(b1, static_cast<float>(t));
    const float bc2 = 1.f - std::pow(b2, static_cast<float>(t));
    for (size_t i = 0; i < w.size(); ++i) {
      m[i] = b1 * m[i] + (1.f - b1) * g[i];
      v[i] = b2 * v[i] + (1.f - b2) * g[i] * g[i];
      w[i] -= lr * (m[i] / bc1) / (std::sqrt(v[i] / bc2) + 1e-8f);
    }
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: gcn_train JOB_DIR\n";
    return 2;
  }
  const std::string root = argv[1];
  try {
    Meta meta;
    {
      std::ifstream in(root + "/meta.txt");
      if (!in) {
        throw std::runtime_error("missing meta.txt");
      }
      in >> meta.n >> meta.d >> meta.e >> meta.n_pos >> meta.n_mark >> meta.max_epochs >> meta.patience >>
          meta.seed >> meta.hidden >> meta.latent >> meta.loss >> meta.lr;
    }
    if (meta.n <= 0 || meta.d <= 0) {
      throw std::runtime_error("empty graph");
    }
    meta.hidden = std::min(meta.hidden, std::max(4, meta.d * 2));
    meta.latent = std::min(meta.latent, meta.hidden);
    auto indptr = read_vec<int>(root + "/indptr.i32", meta.n + 1);
    auto indices = read_vec<int>(root + "/indices.i32", meta.e);
    auto values = read_vec<float>(root + "/values.f32", meta.e);
    auto features = read_vec<float>(root + "/features.f32", meta.n * meta.d);
    auto pos = read_vec<int>(root + "/pos.i32", std::max(meta.n_pos, 0) * 2);
    auto mark = read_vec<int>(root + "/mark.i32", std::max(meta.n_mark, 0) * 2);

    std::mt19937 rng(static_cast<unsigned>(meta.seed));
    std::vector<int> order(meta.n_pos);
    for (int i = 0; i < meta.n_pos; ++i) {
      order[i] = i;
    }
    std::shuffle(order.begin(), order.end(), rng);
    int n_val = 0;
    int n_train = meta.n_pos;
    if (meta.n_pos >= 8) {
      n_val = std::max(1, meta.n_pos / 5);
      n_train = meta.n_pos - n_val;
    }
    std::vector<int> train_i, train_j, val_i, val_j;
    for (int t = 0; t < n_train; ++t) {
      train_i.push_back(pos[order[t] * 2]);
      train_j.push_back(pos[order[t] * 2 + 1]);
    }
    for (int t = n_train; t < n_train + n_val; ++t) {
      val_i.push_back(pos[order[t] * 2]);
      val_j.push_back(pos[order[t] * 2 + 1]);
    }
    if (n_val == 0) {
      val_i = train_i;
      val_j = train_j;
      n_val = n_train;
    }
    std::uniform_int_distribution<int> node_draw(0, meta.n - 1);
    auto draw_neg = [&](int count, std::vector<int>& left, std::vector<int>& right) {
      left.resize(count);
      right.resize(count);
      for (int i = 0; i < count; ++i) {
        int a = node_draw(rng);
        int b = node_draw(rng);
        if (a == b) {
          b = (b + 1) % meta.n;
        }
        left[i] = a;
        right[i] = b;
      }
    };
    std::vector<int> val_neg_i, val_neg_j;
    draw_neg(std::max(n_val * 2, 1), val_neg_i, val_neg_j);

    std::normal_distribution<float> normal(0.f, 1.f);
    auto init = [&](int rows, int cols) {
      std::vector<float> w(rows * cols);
      const float scale = std::sqrt(2.f / static_cast<float>(std::max(rows, 1)));
      for (float& value : w) {
        value = normal(rng) * scale;
      }
      return w;
    };
    std::vector<float> w0 = init(meta.d, meta.hidden);
    std::vector<float> w1 = init(meta.hidden, meta.latent);
    Adam opt(meta.lr, meta.hidden, meta.d, meta.latent);
    std::vector<float> best0 = w0;
    std::vector<float> best1 = w1;
    float best_val = 1e30f;
    int stall = 0;
    int epochs = 0;
    float last_train = 0.f;
    const float marker_weight = meta.loss == 1 ? 1.f : meta.loss == 2 ? 2.f : 0.f;

    std::vector<float> xw(meta.n * meta.hidden);
    std::vector<float> pre_h(meta.n * meta.hidden);
    std::vector<float> hidden_state(meta.n * meta.hidden);
    std::vector<float> hw(meta.n * meta.latent);
    std::vector<float> z(meta.n * meta.latent);
    std::vector<float> grad_z(meta.n * meta.latent);
    std::vector<float> grad_u(meta.n * meta.latent);
    std::vector<float> grad_h(meta.n * meta.hidden);
    std::vector<float> grad_pre(meta.n * meta.hidden);
    std::vector<float> grad_xw(meta.n * meta.hidden);
    std::vector<float> grad_w0(meta.d * meta.hidden);
    std::vector<float> grad_w1(meta.hidden * meta.latent);

    auto embed = [&](const std::vector<float>& local0, const std::vector<float>& local1) {
      for (int i = 0; i < meta.n; ++i) {
        for (int h = 0; h < meta.hidden; ++h) {
          float sum = 0.f;
          for (int k = 0; k < meta.d; ++k) {
            sum += features[i * meta.d + k] * local0[k * meta.hidden + h];
          }
          xw[i * meta.hidden + h] = sum;
        }
      }
      spmm(indptr, indices, values, xw, meta.n, meta.hidden, meta.hidden, pre_h);
      for (int i = 0; i < meta.n * meta.hidden; ++i) {
        hidden_state[i] = pre_h[i] > 0.f ? pre_h[i] : 0.01f * pre_h[i];
      }
      for (int i = 0; i < meta.n; ++i) {
        for (int k = 0; k < meta.latent; ++k) {
          float sum = 0.f;
          for (int h = 0; h < meta.hidden; ++h) {
            sum += hidden_state[i * meta.hidden + h] * local1[h * meta.latent + k];
          }
          hw[i * meta.latent + k] = sum;
        }
      }
      spmm(indptr, indices, values, hw, meta.n, meta.latent, meta.latent, z);
    };

    auto pair_loss = [&](const std::vector<int>& pi, const std::vector<int>& pj, const std::vector<int>& ni,
                         const std::vector<int>& nj, bool with_marker) {
      std::fill(grad_z.begin(), grad_z.end(), 0.f);
      const int npair = static_cast<int>(pi.size());
      const int nneg = static_cast<int>(ni.size());
      double pos_sum = 0.0;
      double neg_sum = 0.0;
      for (int t = 0; t < npair; ++t) {
        const float logit = dot_row(z, pi[t], pj[t], meta.latent);
        const float positive = std::max(logit, 0.f);
        pos_sum += positive - logit + std::log1p(std::exp(-std::fabs(logit)));
        const float prob = 1.f / (1.f + std::exp(-logit));
        const float g = 0.5f * (prob - 1.f) / static_cast<float>(std::max(npair, 1));
        add_bilinear(grad_z, z, pi[t], pj[t], g, meta.latent);
      }
      for (int t = 0; t < nneg; ++t) {
        const float logit = dot_row(z, ni[t], nj[t], meta.latent);
        const float positive = std::max(logit, 0.f);
        neg_sum += positive + std::log1p(std::exp(-std::fabs(logit)));
        const float prob = 1.f / (1.f + std::exp(-logit));
        const float g = 0.5f * prob / static_cast<float>(std::max(nneg, 1));
        add_bilinear(grad_z, z, ni[t], nj[t], g, meta.latent);
      }
      double total = 0.5 * ((npair ? pos_sum / npair : 0.0) + (nneg ? neg_sum / nneg : 0.0));
      if (with_marker && marker_weight > 0.f && meta.n_mark > 0) {
        double extra = 0.0;
        for (int t = 0; t < meta.n_mark; ++t) {
          const int a = mark[t * 2];
          const int b = mark[t * 2 + 1];
          float dist = 0.f;
          for (int k = 0; k < meta.latent; ++k) {
            const float delta = z[a * meta.latent + k] - z[b * meta.latent + k];
            dist += delta * delta;
          }
          const float energy = std::exp(-0.5f * dist);
          extra += energy;
          const float coeff = marker_weight * energy / static_cast<float>(meta.n_mark);
          for (int k = 0; k < meta.latent; ++k) {
            const float za = z[a * meta.latent + k];
            const float zb = z[b * meta.latent + k];
            grad_z[a * meta.latent + k] += coeff * (zb - za);
            grad_z[b * meta.latent + k] += coeff * (za - zb);
          }
        }
        total += marker_weight * (extra / meta.n_mark);
      }
      return static_cast<float>(total);
    };

    for (int epoch = 0; epoch < meta.max_epochs; ++epoch) {
      opt.t = epoch + 1;
      epochs = epoch + 1;
      embed(w0, w1);
      std::vector<int> use_i = train_i;
      std::vector<int> use_j = train_j;
      if (meta.loss == 3 && !train_i.empty()) {
        use_i.clear();
        use_j.clear();
        std::uniform_int_distribution<int> pick(0, static_cast<int>(train_i.size()) - 1);
        const int take = std::min(512, static_cast<int>(train_i.size()));
        for (int t = 0; t < take; ++t) {
          const int id = pick(rng);
          use_i.push_back(train_i[id]);
          use_j.push_back(train_j[id]);
        }
      }
      std::vector<int> neg_i, neg_j;
      draw_neg(std::max(static_cast<int>(use_i.size()) * 2, 1), neg_i, neg_j);
      last_train = pair_loss(use_i, use_j, neg_i, neg_j, true);

      std::fill(grad_u.begin(), grad_u.end(), 0.f);
      for (int i = 0; i < meta.n; ++i) {
        for (int p = indptr[i]; p < indptr[i + 1]; ++p) {
          const int j = indices[p];
          const float a = values[p];
          for (int k = 0; k < meta.latent; ++k) {
            grad_u[j * meta.latent + k] += a * grad_z[i * meta.latent + k];
          }
        }
      }
      std::fill(grad_w1.begin(), grad_w1.end(), 0.f);
      std::fill(grad_h.begin(), grad_h.end(), 0.f);
      for (int i = 0; i < meta.n; ++i) {
        for (int h = 0; h < meta.hidden; ++h) {
          for (int k = 0; k < meta.latent; ++k) {
            grad_w1[h * meta.latent + k] += hidden_state[i * meta.hidden + h] * grad_u[i * meta.latent + k];
            grad_h[i * meta.hidden + h] += grad_u[i * meta.latent + k] * w1[h * meta.latent + k];
          }
        }
      }
      for (int i = 0; i < meta.n * meta.hidden; ++i) {
        grad_pre[i] = pre_h[i] > 0.f ? grad_h[i] : 0.01f * grad_h[i];
      }
      std::fill(grad_xw.begin(), grad_xw.end(), 0.f);
      for (int i = 0; i < meta.n; ++i) {
        for (int p = indptr[i]; p < indptr[i + 1]; ++p) {
          const int j = indices[p];
          const float a = values[p];
          for (int h = 0; h < meta.hidden; ++h) {
            grad_xw[j * meta.hidden + h] += a * grad_pre[i * meta.hidden + h];
          }
        }
      }
      std::fill(grad_w0.begin(), grad_w0.end(), 0.f);
      for (int i = 0; i < meta.n; ++i) {
        for (int k = 0; k < meta.d; ++k) {
          for (int h = 0; h < meta.hidden; ++h) {
            grad_w0[k * meta.hidden + h] += features[i * meta.d + k] * grad_xw[i * meta.hidden + h];
          }
        }
      }
      opt.apply(w1, opt.m1, opt.v1, grad_w1);
      opt.apply(w0, opt.m0, opt.v0, grad_w0);

      embed(w0, w1);
      const float val = pair_loss(val_i, val_j, val_neg_i, val_neg_j, false);
      if (val + 1e-5f < best_val) {
        best_val = val;
        best0 = w0;
        best1 = w1;
        stall = 0;
      } else {
        ++stall;
        if (stall >= meta.patience) {
          break;
        }
      }
    }
    embed(best0, best1);
    {
      std::ofstream out(root + "/embedding.f32", std::ios::binary);
      out.write(reinterpret_cast<const char*>(z.data()), static_cast<std::streamsize>(z.size() * sizeof(float)));
    }
    {
      std::ofstream out(root + "/result.txt");
      out << epochs << " " << (epochs < meta.max_epochs ? 1 : 0) << " " << best_val << " " << last_train << "\n";
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
  return 0;
}
