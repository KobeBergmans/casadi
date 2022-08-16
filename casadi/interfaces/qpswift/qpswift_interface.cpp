/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
 *                            K.U. Leuven. All rights reserved.
 *    Copyright (C) 2011-2014 Greg Horn
 *
 *    CasADi is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    CasADi is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with CasADi; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */


#include "qpswift_interface.hpp"
#include "casadi/core/casadi_misc.hpp"

using namespace std;
namespace casadi {

  extern "C"
  int CASADI_CONIC_QPSWIFT_EXPORT
  casadi_register_conic_qpswift(Conic::Plugin* plugin) {
    plugin->creator = QpswiftInterface::creator;
    plugin->name = "qpswift";
    plugin->doc = QpswiftInterface::meta_doc.c_str();
    plugin->version = CASADI_VERSION;
    plugin->options = &QpswiftInterface::options_;
    plugin->deserialize = &QpswiftInterface::deserialize;
    return 0;
  }

  extern "C"
  void CASADI_CONIC_QPSWIFT_EXPORT casadi_load_conic_qpswift() {
    Conic::registerPlugin(casadi_register_conic_qpswift);
  }

  QpswiftInterface::QpswiftInterface(const std::string& name,
                                   const std::map<std::string, Sparsity>& st)
    : Conic(name, st) {

    has_refcount_ = true;
  }

  QpswiftInterface::~QpswiftInterface() {
    clear_mem();
  }

  const Options QpswiftInterface::options_
  = {{&Conic::options_},
     {{"maxit",
       {OT_INT,
        "Maximum iterations."}},
      {"reltol",
       {OT_DOUBLE,
        "Relative tolerance."}},
      {"abstol",
       {OT_DOUBLE,
        "Absolute tolerance."}},
      {"sigma",
       {OT_DOUBLE,
        "Desired sigma value."}},
      {"verbose",
       {OT_INT,
        "Printing level."}}
     }
  };

  void QpswiftInterface::init(const Dict& opts) {
    // Initialize the base classes
    Conic::init(opts);
    
    maxit_ = 0;
    reltol_ = 0.;
    abstol_ = 0.;
    sigma_ = 0.;
    verbose_ = 0;

    // Read options
    for (auto&& op : opts) {
      if (op.first=="maxit") {
        int temp = op.second;
        maxit_ = static_cast<qp_int>(temp);
      } else if (op.first=="reltol") {
        reltol_ = op.second;
      } else if (op.first=="abstol") {
        abstol_ = op.second;
      } else if (op.first=="sigma") {
        sigma_ = op.second;
      } else if (op.first=="verbose") {
        int temp = op.second;
        verbose_ = static_cast<qp_int>(temp);
      }
    }

    alloc_iw(na_); // casadi_trans
  }

  int QpswiftInterface::init_mem(void* mem) const {
    if (Conic::init_mem(mem)) return 1;

    return 0;
  }

  inline const char* return_status_string(casadi_int status) {
    return "Unknown";
  }

  int QpswiftInterface::
  solve(const double** arg, double** res, casadi_int* iw, double* w, void* mem) const {
    auto m = static_cast<QpswiftMemory*>(mem);

    // Null pointer
    qp_int *null_int = NULL;
    qp_real *null_real = NULL;

    // QP var
    QP *qp_prob;

    // Get constraints
    const double *lbx = arg[CONIC_LBX];
    const double *lba = arg[CONIC_LBA];
    const double *ubx = arg[CONIC_UBX];
    const double *uba = arg[CONIC_UBA];
    std::vector<qp_real> lbx_data = vector_static_cast<qp_real>(std::vector<double>(lbx, lbx+nx_));
    std::vector<qp_real> lba_data = vector_static_cast<qp_real>(std::vector<double>(lba, lba+na_));
    std::vector<qp_real> ubx_data = vector_static_cast<qp_real>(std::vector<double>(ubx, ubx+nx_));
    std::vector<qp_real> uba_data = vector_static_cast<qp_real>(std::vector<double>(uba, uba+na_));

    // Check for equality constraints in A
    // TODO(@KobeBergmans): Make this assignment more optimal.
    std::vector<casadi_int> equality_constr_A = std::vector<casadi_int>(na_);
    casadi_int constr_index = 0;
    for (casadi_int i = 0; i < na_; ++i) {
      if (lba_data[i] == uba_data[i]) equality_constr_A[constr_index++] = i;
    }
    equality_constr_A.resize(constr_index);

    uout() << "Equality constraints in A on indices: " << equality_constr_A << std::endl;

    // Check for equality constraints in x
    // TODO(@KobeBergmans): Make this assignment more optimal.
    std::vector<casadi_int> equality_constr_x = std::vector<casadi_int>(nx_);
    constr_index = 0;
    for (casadi_int i = 0; i < nx_; ++i) {
      if (lbx_data[i] == ubx_data[i]) equality_constr_x[constr_index++] = i;
    }
    equality_constr_x.resize(constr_index);

    uout() << "Equality constraints in x on indices: " << equality_constr_x << std::endl;

    // Number of desiscion vars and constraints
    qp_int nc = nx_;
    qp_int mc = 2*(nx_+na_-equality_constr_A.size()-equality_constr_x.size()) ; // 2 times because we have a double inequality
    qp_int pc = equality_constr_A.size() + equality_constr_x.size();

    // TODO(@KobeBergmans): We should probably use more casadi built ins here instead of the vector operations.
    // TODO(@KobeBergmans): We should also put the vectors in the memory object instead of vector objects for codegen

    // P sparsity
    std::vector<qp_int> Pjc_vec = vector_static_cast<qp_int>(H_.get_colind());
    std::vector<qp_int> Pir_vec = vector_static_cast<qp_int>(H_.get_row());
    qp_int* Pjc = Pjc_vec.data();
    qp_int* Pir = Pir_vec.data();

    uout() << "H/P sparsity: " << std::endl;
    H_.spy(uout());

    // P data
    const double *h_prob = arg[CONIC_H];
    std::vector<qp_real> Ppr_vec = vector_static_cast<qp_real>(std::vector<double>(h_prob, h_prob+H_.nnz()));
    qp_real* Ppr = Ppr_vec.data();

    uout() << "H/P data: " << Ppr_vec << std::endl;

    // A Sparsity, Ax part
    Sparsity A_trans = A_.T(); // Get transpose because we need to copy columns
    const double *a_prob = arg[CONIC_A];
    const casadi_int *A_trans_colind, *A_trans_row;
    Sparsity A_sp_a_trans;
    std::vector<qp_real> new_data;
    std::vector<casadi_int> new_row, new_colind;

    std::vector<qp_real> a_prob_trans_vec = std::vector<qp_real>(A_.nnz(), 0);
    double* a_prob_trans = a_prob_trans_vec.data();
    casadi_trans(a_prob, A_, a_prob_trans, A_trans, iw);
    if (equality_constr_A.size() > 0) {
      A_trans_colind = A_trans.colind();
      A_trans_row = A_trans.row();

      new_colind = std::vector<casadi_int>(equality_constr_A.size()+1, 0);
      new_data = std::vector<qp_real>(equality_constr_x.size()+A_.nnz(), 1.);
      constr_index = 0;
      for (casadi_int i = 0; i < A_.size1(); ++i) { // Loop over all the constraints
        if (equality_constr_A[constr_index] == i) {
          new_row.insert(new_row.begin()+new_colind[constr_index], A_trans_row, A_trans_row + *(A_trans_colind+1) - *A_trans_colind);
          new_data.insert(new_data.begin()+new_colind[constr_index]+equality_constr_x.size(), a_prob_trans, a_prob_trans + *(A_trans_colind+1) - *A_trans_colind);
          new_colind[constr_index+1] = new_colind[constr_index] + *(A_trans_colind+1) - *A_trans_colind;
          constr_index++;
          if (constr_index == equality_constr_A.size()) break;
        } 

        A_trans_row += *(A_trans_colind+1) - *A_trans_colind;
        a_prob_trans += *(A_trans_colind+1) - *A_trans_colind;
        A_trans_colind++;
      }
      A_sp_a_trans = Sparsity(nx_, equality_constr_A.size(), new_colind, new_row);
    } else {
      A_sp_a_trans = Sparsity(0,0);
    }
    
    // A Sparsity, x part
    Sparsity A_sp_trans;
    if (equality_constr_x.size() > 0) {
      A_sp_trans = Sparsity(nx_, equality_constr_x.size());
      for (casadi_int i = 0; i < equality_constr_x.size(); ++i) A_sp_trans.add_nz(equality_constr_x[i], i);
    } else {
      A_sp_trans = Sparsity(0,0);
    }

    // A Sparsity
    Sparsity A_sp;
    qp_int *Ajc = NULL;
    qp_int *Air = NULL;
    if (equality_constr_x.size() > 0 || equality_constr_A.size() > 0) {
      A_sp_trans.appendColumns(A_sp_a_trans);
      A_sp = A_sp_trans.T();
      std::vector<qp_int> Ajc_vec = vector_static_cast<qp_int>(A_sp.get_colind());
      std::vector<qp_int> Air_vec = vector_static_cast<qp_int>(A_sp.get_row());
      Ajc = Ajc_vec.data();
      Air = Air_vec.data();

      uout() << "equality constr sparsity: " << std::endl;
      A_sp.spy(uout());
    }    

    // A Data
    std::vector<qp_real> Apr_vec = std::vector<qp_real>(A_sp.nnz(), 0);
    casadi_trans(new_data.data(), A_sp_trans, Apr_vec.data(), A_sp, iw);
    qp_real* Apr = Apr_vec.data();
    uout() << "equality constr data: " << Apr_vec << std::endl;
    uout() << "Apr: " << Apr << std::endl;

    if (equality_constr_A.size() == 0 && equality_constr_x.size() == 0) {
      qp_real* Apr = NULL;
    }

    uout() << "Apr: " << Apr << std::endl;

    // G sparsity, Ax part
    Sparsity G_sp_g_trans;
    if (equality_constr_A.size() != na_) {
      a_prob = arg[CONIC_A];
      A_trans_colind = A_trans.colind();
      A_trans_row = A_trans.row();
      a_prob_trans = a_prob_trans_vec.data();

      new_colind = std::vector<casadi_int>(na_ - equality_constr_A.size()+1, 0);
      new_row = std::vector<casadi_int>();
      new_data = std::vector<qp_real>(2*(nx_-equality_constr_x.size() + A_.nnz()-A_sp_a_trans.nnz()), 1.);
      fill_n(new_data.begin()+nx_-equality_constr_x.size(), nx_-equality_constr_x.size(), -1.);
      constr_index = 0;
      casadi_int colind_index = 0;
      for (casadi_int i = 0; i < A_.size1(); ++i) { // Loop over all the constraints
        if (equality_constr_A.size() == 0 || equality_constr_A[constr_index] != i) {
          new_row.insert(new_row.begin()+new_colind[colind_index], A_trans_row, A_trans_row + *(A_trans_colind+1) - *A_trans_colind);
          
          casadi_copy(a_prob_trans, *(A_trans_colind+1) - *A_trans_colind, new_data.data()+new_colind[colind_index]+2*(nx_-equality_constr_x.size()));
          casadi_copy(a_prob_trans, *(A_trans_colind+1) - *A_trans_colind, new_data.data()+new_colind[colind_index]+2*(nx_-equality_constr_x.size())+A_.nnz()-A_sp_a_trans.nnz());
          transform(new_data.begin()+new_colind[colind_index]+2*(nx_-equality_constr_x.size())+A_.nnz()-A_sp_a_trans.nnz(), new_data.begin()+new_colind[colind_index]+2*(nx_-equality_constr_x.size())+A_.nnz()-A_sp_a_trans.nnz()+ *(A_trans_colind+1) - *A_trans_colind, new_data.begin()+new_colind[colind_index]+2*(nx_-equality_constr_x.size())+A_.nnz()-A_sp_a_trans.nnz(), std::negate<qp_real>());
          
          new_colind[colind_index+1] = new_colind[colind_index] + *(A_trans_colind+1) - *A_trans_colind;
          colind_index++;
        } else {
          constr_index++;
        }

        A_trans_row += *(A_trans_colind+1) - *A_trans_colind;
        a_prob_trans += *(A_trans_colind+1) - *A_trans_colind;
        A_trans_colind++;
      }
      G_sp_g_trans = Sparsity(nx_, na_ - equality_constr_A.size(), new_colind, new_row);
    } else {
      G_sp_g_trans = Sparsity(0,0);
    }

    // G sparsity, x part
    Sparsity G_sp_trans;
    if (equality_constr_x.size() != nx_) {
      Sparsity G_sp_x_trans = Sparsity(nx_, nx_ - equality_constr_x.size());
      constr_index = 0;
      casadi_int row_ind = 0;
      for (casadi_int i = 0; i < nx_; ++i) {
        if (equality_constr_x.size() == 0 || i != equality_constr_x[constr_index]) {
          G_sp_x_trans.add_nz(i, row_ind);
          row_ind++;
        } else {
          constr_index++;
        }
      }
      G_sp_trans = Sparsity(G_sp_x_trans);
      G_sp_trans.appendColumns(G_sp_x_trans);
    } else {
      G_sp_trans = Sparsity(0,0);
    }

    // G sparsity
    qp_int *Gjc = NULL;
    qp_int *Gir = NULL;
    Sparsity G_sp;
    if (equality_constr_A.size() != na_ || equality_constr_x.size() != nx_) {
      G_sp_trans.appendColumns(G_sp_g_trans);
      G_sp_trans.appendColumns(G_sp_g_trans);
      G_sp = G_sp_trans.T();
      std::vector<qp_int> Gjc_vec = vector_static_cast<qp_int>(G_sp.get_colind());
      std::vector<qp_int> Gir_vec = vector_static_cast<qp_int>(G_sp.get_row());
      Gjc = Gjc_vec.data();
      Gir = Gir_vec.data();

      uout() << "A/G sparsity: " << std::endl;
      G_sp.spy(uout());
    }
    
    // G data
    qp_real* Gpr = NULL;
    if (equality_constr_A.size() != na_ || equality_constr_x.size() != nx_) {
      std::vector<qp_real> Gpr_vec = std::vector<qp_real>(nx_ - equality_constr_x.size() + G_sp.nnz(), 0); // Make it big enough for reuse
      casadi_trans(new_data.data(), G_sp_trans, Gpr_vec.data(), G_sp, iw);
      Gpr = Gpr_vec.data();
      uout() << "A/G data: " << Gpr_vec << std::endl;
    }

    // c data
    qp_real* c = NULL;
    if (arg[CONIC_G]) {
      const double *g = arg[CONIC_G];
      std::vector<qp_real> c_vec = vector_static_cast<qp_real>(std::vector<double>(g, g+nx_));
      c = c_vec.data();

      uout() << "g/C data: " << c_vec << std::endl;
    }

    // h Data
    qp_real* h = NULL;
    if (equality_constr_A.size() != na_ || equality_constr_x.size() != nx_) {
      std::vector<qp_real> h_vec(2*(nx_+na_-equality_constr_x.size()-equality_constr_A.size()), 0.);
      casadi_int h_index = 0;
      constr_index = 0;
      for (casadi_int i = 0; i < nx_; ++i) {
        if (equality_constr_x.size() == 0 || i != equality_constr_x[constr_index]) {
          h_vec[h_index] = ubx_data[i];
          h_vec[h_index+nx_-equality_constr_x.size()] = -lbx_data[i];
          h_index++;
        } else {
          constr_index++;
        }
      }
      
      h_index = 0;
      constr_index = 0;
      for (casadi_int i = 0; i < na_; ++i) {
        if (equality_constr_A.size() == 0 || i != equality_constr_A[constr_index]) {
          h_vec[h_index+2*(nx_-equality_constr_x.size())] = uba_data[i];
          h_vec[h_index+2*(nx_-equality_constr_x.size())+na_-equality_constr_A.size()] = -lba_data[i];
          h_index++;
        } else {
          constr_index++;
        }
      }

      h = h_vec.data();
      uout() << "bnds/h data: " << h_vec << std::endl;
    }

    // b Data
    qp_real* b = NULL;
    if (equality_constr_A.size() > 0 || equality_constr_x.size() > 0) {
      std::vector<qp_real> b_vec(equality_constr_x.size()+equality_constr_A.size(), 0.);
      casadi_int h_index = 0;
      if (equality_constr_x.size() > 0) {
        constr_index = 0;
        for (casadi_int i = 0; i < nx_; ++i) {
          if (i == equality_constr_x[constr_index]) {
              b_vec[h_index] = ubx_data[i];
              h_index++;
              constr_index++;
          }
        }
      }
      
      if (equality_constr_A.size() > 0) {
        constr_index = 0;
        for (casadi_int i = 0; i < na_; ++i) {
          if (i == equality_constr_A[constr_index]) {
              b_vec[h_index] = uba_data[i];
              h_index++;
              constr_index++;
          }
        }
      }
    
      b = b_vec.data();
      uout() << "equality bnds: " << b_vec << std::endl;
    }

    // Get solver
    uout() << "Problem sizes: " << nc << ", " << mc << ", " << pc << std::endl;
    uout() << "Data locations: " << std::endl;
    uout() << "P: " << Pjc << ", " << Pir << ", " << Ppr << std::endl;
    uout() << "A: " << Ajc << ", " << Air << ", " << Apr << std::endl;
    uout() << "G: " << Gjc << ", " << Gir << ", " << Gpr << std::endl;
    uout() << "c, h, b: " << c << ", " << h  << ", " << b << std::endl;
    qp_prob = QP_SETUP(nc, mc, pc, Pjc, Pir, Ppr, Ajc, Air, Apr, Gjc, Gir, Gpr, c, h, b, 
                    0, NULL);

    // Change solver settings
    if (maxit_ != 0) qp_prob->options->maxit = maxit_;
    if (reltol_ != 0.) qp_prob->options->reltol = reltol_;
    if (abstol_ != 0.) qp_prob->options->abstol = abstol_;
    if (sigma_ != 0.) qp_prob->options->sigma = sigma_;
    if (verbose_ != 0) qp_prob->options->verbose = verbose_;

    // Solve QP
    qp_int exit_code = QP_SOLVE(qp_prob);
    if (exit_code == QP_OPTIMAL) {
      uout() << "qpswift succeeded" << std::endl;
      m->success = true;
      m->unified_return_status = SOLVER_RET_SUCCESS;
    } else if (exit_code == QP_FATAL || exit_code == QP_KKTFAIL || exit_code == QP_MAXIT) {
      uout() << "qpswift failed due to " << exit_code << std::endl;
      m->success = false;
      m->unified_return_status = SOLVER_RET_INFEASIBLE;
      return 1;
    } 

    // Copy output x
    casadi_copy(qp_prob->x, nx_, res[CONIC_X]);

    // Copy output lam_x
    double* lam_x = res[CONIC_LAM_X];
    double* z_out = qp_prob->z;
    double* z_out_dual = qp_prob->z + nx_ - equality_constr_x.size();
    double* y_out = qp_prob->y;

    constr_index = 0;
    for (casadi_int i = 0; i < nx_; ++i) {
      if (i == equality_constr_x[constr_index]) {
        *lam_x++ = *y_out++;
        constr_index++;
      } else {
        *lam_x = *z_out++;
        *lam_x++ -= *z_out_dual++;
      } 
    }

    // Copy output lam_a
    double* lam_a = res[CONIC_LAM_A];
    z_out = qp_prob->z+2*(nx_ - equality_constr_x.size());
    z_out_dual = qp_prob->z+2*(nx_ - equality_constr_x.size())+na_-equality_constr_A.size();
    y_out = qp_prob->y+equality_constr_x.size();

    constr_index = 0;
    for (casadi_int i = 0; i < na_; ++i) {
      if (i == equality_constr_A[constr_index]) {
        *lam_a++ = *y_out++;
        constr_index++;
      } else {
        *lam_a = *z_out++;
        *lam_a++ -= *z_out_dual++;
      } 
    }

    // Copy cost
    if (res[CONIC_COST]) *res[CONIC_COST] = qp_prob->stats->fval;

    // Copy stats
    m->tsetup = qp_prob->stats->tsetup;
    m->tsolve = qp_prob->stats->tsolve;
    m->kkt_time = qp_prob->stats->kkt_time;
    m->ldl_numeric = qp_prob->stats->ldl_numeric;
    m->iter_count = static_cast<int>(qp_prob->stats->IterationCount);

    // Cleanup
    QP_CLEANUP(qp_prob);

    return 0;
  }

  void QpswiftInterface::codegen_free_mem(CodeGenerator& g) const {
    g << "qpswift_cleanup(" + codegen_mem(g) + ");\n";
  }

  void QpswiftInterface::codegen_init_mem(CodeGenerator& g) const {
    Sparsity Asp = vertcat(Sparsity::diag(nx_), A_);
    casadi_int dummy_size = max(nx_+na_, max(Asp.nnz(), H_.nnz()));

    g.local("A", "csc");
    g.local("dummy[" + str(dummy_size) + "]", "casadi_real");
    g << g.clear("dummy", dummy_size) << "\n";

    g.constant_copy("A_row", Asp.get_row(), "c_int");
    g.constant_copy("A_colind", Asp.get_colind(), "c_int");
    g.constant_copy("H_row", H_.get_row(), "c_int");
    g.constant_copy("H_colind", H_.get_colind(), "c_int");

    g.local("A", "csc");
    g << "A.m = " << nx_ + na_ << ";\n";
    g << "A.n = " << nx_ << ";\n";
    g << "A.nz = " << nnzA_ << ";\n";
    g << "A.nzmax = " << nnzA_ << ";\n";
    g << "A.x = dummy;\n";
    g << "A.i = A_row;\n";
    g << "A.p = A_colind;\n";

    g.local("H", "csc");
    g << "H.m = " << nx_ << ";\n";
    g << "H.n = " << nx_ << ";\n";
    g << "H.nz = " << H_.nnz() << ";\n";
    g << "H.nzmax = " << H_.nnz() << ";\n";
    g << "H.x = dummy;\n";
    g << "H.i = H_row;\n";
    g << "H.p = H_colind;\n";

    g.local("data", "QPSWIFTData");
    g << "data.n = " << nx_ << ";\n";
    g << "data.m = " << nx_ + na_ << ";\n";
    g << "data.P = &H;\n";
    g << "data.q = dummy;\n";
    g << "data.A = &A;\n";
    g << "data.l = dummy;\n";
    g << "data.u = dummy;\n";

    // g.local("settings", "QPSWIFTSettings");
    // g << "qpswift_set_default_settings(&settings);\n";
    // g << "settings.rho = " << settings_.rho << ";\n";
    // g << "settings.sigma = " << settings_.sigma << ";\n";
    // g << "settings.scaling = " << settings_.scaling << ";\n";
    // g << "settings.adaptive_rho = " << settings_.adaptive_rho << ";\n";
    // g << "settings.adaptive_rho_interval = " << settings_.adaptive_rho_interval << ";\n";
    // g << "settings.adaptive_rho_tolerance = " << settings_.adaptive_rho_tolerance << ";\n";
    //g << "settings.adaptive_rho_fraction = " << settings_.adaptive_rho_fraction << ";\n";
    // g << "settings.max_iter = " << settings_.max_iter << ";\n";
    // g << "settings.eps_abs = " << settings_.eps_abs << ";\n";
    // g << "settings.eps_rel = " << settings_.eps_rel << ";\n";
    // g << "settings.eps_prim_inf = " << settings_.eps_prim_inf << ";\n";
    // g << "settings.eps_dual_inf = " << settings_.eps_dual_inf << ";\n";
    // g << "settings.alpha = " << settings_.alpha << ";\n";
    // g << "settings.delta = " << settings_.delta << ";\n";
    // g << "settings.polish = " << settings_.polish << ";\n";
    // g << "settings.polish_refine_iter = " << settings_.polish_refine_iter << ";\n";
    // g << "settings.verbose = " << settings_.verbose << ";\n";
    // g << "settings.scaled_termination = " << settings_.scaled_termination << ";\n";
    // g << "settings.check_termination = " << settings_.check_termination << ";\n";
    // g << "settings.warm_start = " << settings_.warm_start << ";\n";
    //g << "settings.time_limit = " << settings_.time_limit << ";\n";

    g << codegen_mem(g) + " = qpswift_setup(&data, &settings);\n";
    g << "return 0;\n";
  }

  void QpswiftInterface::codegen_body(CodeGenerator& g) const {
    g.add_include("qpswift/qpswift.h");
    g.add_auxiliary(CodeGenerator::AUX_INF);

    g.local("work", "QPSWIFTWorkspace", "*");
    g.init_local("work", codegen_mem(g));

    g.comment("Set objective");
    g.copy_default(g.arg(CONIC_G), nx_, "w", "0", false);
    g << "if (qpswift_update_lin_cost(work, w)) return 1;\n";

    g.comment("Set bounds");
    g.copy_default(g.arg(CONIC_LBX), nx_, "w", "-casadi_inf", false);
    g.copy_default(g.arg(CONIC_LBA), na_, "w+"+str(nx_), "-casadi_inf", false);
    g.copy_default(g.arg(CONIC_UBX), nx_, "w+"+str(nx_+na_), "casadi_inf", false);
    g.copy_default(g.arg(CONIC_UBA), na_, "w+"+str(2*nx_+na_), "casadi_inf", false);
    g << "if (qpswift_update_bounds(work, w, w+" + str(nx_+na_)+ ")) return 1;\n";

    g.comment("Project Hessian");
    g << g.tri_project(g.arg(CONIC_H), H_, "w", false);

    g.comment("Get constraint matrix");
    std::string A_colind = g.constant(A_.get_colind());
    g.local("offset", "casadi_int");
    g.local("n", "casadi_int");
    g.local("i", "casadi_int");
    g << "offset = 0;\n";
    g << "for (i=0; i< " << nx_ << "; ++i) {\n";
    g << "w[" + str(nnzHupp_) + "+offset] = 1;\n";
    g << "offset++;\n";
    g << "n = " + A_colind + "[i+1]-" + A_colind + "[i];\n";
    g << "casadi_copy(" << g.arg(CONIC_A) << "+" + A_colind + "[i], n, "
         "w+offset+" + str(nnzHupp_) + ");\n";
    g << "offset+= n;\n";
    g << "}\n";

    g.comment("Pass Hessian and constraint matrices");
    g << "if (qpswift_update_P_A(work, w, 0, " + str(nnzHupp_) + ", w+" + str(nnzHupp_) +
         ", 0, " + str(nnzA_) + ")) return 1;\n";

    g << "if (qpswift_warm_start_x(work, " + g.arg(CONIC_X0) + ")) return 1;\n";
    g.copy_default(g.arg(CONIC_LAM_X0), nx_, "w", "0", false);
    g.copy_default(g.arg(CONIC_LAM_A0), na_, "w+"+str(nx_), "0", false);
    g << "if (qpswift_warm_start_y(work, w)) return 1;\n";

    g << "if (qpswift_solve(work)) return 1;\n";

    g.copy_check("&work->info->obj_val", 1, g.res(CONIC_COST), false, true);
    g.copy_check("work->solution->x", nx_, g.res(CONIC_X), false, true);
    g.copy_check("work->solution->y", nx_, g.res(CONIC_LAM_X), false, true);
    g.copy_check("work->solution->y+" + str(nx_), na_, g.res(CONIC_LAM_A), false, true);

    g << "if (work->info->status_val != QPSWIFT_SOLVED) return 1;\n";
  }

  Dict QpswiftInterface::get_stats(void* mem) const {
    Dict stats = Conic::get_stats(mem);
    for (auto&& op : stats_) {
      stats[op.first] = op.second;
    }
    
    return stats;
  }

  QpswiftMemory::QpswiftMemory() {
  }

  QpswiftMemory::~QpswiftMemory() {
  }

  QpswiftInterface::QpswiftInterface(DeserializingStream& s) : Conic(s) {
    s.version("QpswiftInterface", 1);
    s.unpack("QpswiftInterface::nnzHupp", nnzHupp_);
    s.unpack("QpswiftInterface::nnzA", nnzA_);
    s.unpack("QpswiftInterface::warm_start_primal", warm_start_primal_);
    s.unpack("QpswiftInterface::warm_start_dual", warm_start_dual_);
    s.unpack("QpswiftInterface::stats", stats_);
    // s.unpack("QpswiftInterface::settings::rho", settings_.rho);
    // s.unpack("QpswiftInterface::settings::sigma", settings_.sigma);
    // s.unpack("QpswiftInterface::settings::scaling", settings_.scaling);
    // s.unpack("QpswiftInterface::settings::adaptive_rho", settings_.adaptive_rho);
    // s.unpack("QpswiftInterface::settings::adaptive_rho_interval", settings_.adaptive_rho_interval);
    // s.unpack("QpswiftInterface::settings::adaptive_rho_tolerance", settings_.adaptive_rho_tolerance);
    //s.unpack("QpswiftInterface::settings::adaptive_rho_fraction", settings_.adaptive_rho_fraction);
    // s.unpack("QpswiftInterface::settings::max_iter", settings_.max_iter);
    // s.unpack("QpswiftInterface::settings::eps_abs", settings_.eps_abs);
    // s.unpack("QpswiftInterface::settings::eps_rel", settings_.eps_rel);
    // s.unpack("QpswiftInterface::settings::eps_prim_inf", settings_.eps_prim_inf);
    // s.unpack("QpswiftInterface::settings::eps_dual_inf", settings_.eps_dual_inf);
    // s.unpack("QpswiftInterface::settings::alpha", settings_.alpha);
    // s.unpack("QpswiftInterface::settings::delta", settings_.delta);
    // s.unpack("QpswiftInterface::settings::polish", settings_.polish);
    // s.unpack("QpswiftInterface::settings::polish_refine_iter", settings_.polish_refine_iter);
    // s.unpack("QpswiftInterface::settings::verbose", settings_.verbose);
    // s.unpack("QpswiftInterface::settings::scaled_termination", settings_.scaled_termination);
    // s.unpack("QpswiftInterface::settings::check_termination", settings_.check_termination);
    // s.unpack("QpswiftInterface::settings::warm_start", settings_.warm_start);
    //s.unpack("QpswiftInterface::settings::time_limit", settings_.time_limit);
  }

  void QpswiftInterface::serialize_body(SerializingStream &s) const {
    Conic::serialize_body(s);
    s.version("QpswiftInterface", 1);
    s.pack("QpswiftInterface::nnzHupp", nnzHupp_);
    s.pack("QpswiftInterface::nnzA", nnzA_);
    s.pack("QpswiftInterface::warm_start_primal", warm_start_primal_);
    s.pack("QpswiftInterface::warm_start_dual", warm_start_dual_);
    // s.pack("QpswiftInterface::settings::rho", settings_.rho);
    // s.pack("QpswiftInterface::settings::sigma", settings_.sigma);
    // s.pack("QpswiftInterface::settings::scaling", settings_.scaling);
    // s.pack("QpswiftInterface::settings::adaptive_rho", settings_.adaptive_rho);
    // s.pack("QpswiftInterface::settings::adaptive_rho_interval", settings_.adaptive_rho_interval);
    // s.pack("QpswiftInterface::settings::adaptive_rho_tolerance", settings_.adaptive_rho_tolerance);
    //s.pack("QpswiftInterface::settings::adaptive_rho_fraction", settings_.adaptive_rho_fraction);
    // s.pack("QpswiftInterface::settings::max_iter", settings_.max_iter);
    // s.pack("QpswiftInterface::settings::eps_abs", settings_.eps_abs);
    // s.pack("QpswiftInterface::settings::eps_rel", settings_.eps_rel);
    // s.pack("QpswiftInterface::settings::eps_prim_inf", settings_.eps_prim_inf);
    // s.pack("QpswiftInterface::settings::eps_dual_inf", settings_.eps_dual_inf);
    // s.pack("QpswiftInterface::settings::alpha", settings_.alpha);
    // s.pack("QpswiftInterface::settings::delta", settings_.delta);
    // s.pack("QpswiftInterface::settings::polish", settings_.polish);
    // s.pack("QpswiftInterface::settings::polish_refine_iter", settings_.polish_refine_iter);
    // s.pack("QpswiftInterface::settings::verbose", settings_.verbose);
    // s.pack("QpswiftInterface::settings::scaled_termination", settings_.scaled_termination);
    // s.pack("QpswiftInterface::settings::check_termination", settings_.check_termination);
    // s.pack("QpswiftInterface::settings::warm_start", settings_.warm_start);
    //s.pack("QpswiftInterface::settings::time_limit", settings_.time_limit);
  }

} // namespace casadi
