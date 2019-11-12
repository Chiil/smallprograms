#include <iostream>
#include <sstream>
#include <fftw3.h>

float* fftini;
float* fftinj;
float* fftouti;
float* fftoutj;

void save(
        const int itot, const int jtot,
        std::stringstream& ss_iplanf,
        std::stringstream& ss_iplanb,
        std::stringstream& ss_jplanf,
        std::stringstream& ss_jplanb)
{
    int rank = 1;
    int ni[] = {itot};
    int nj[] = {jtot};
    int istride = 1;
    int jstride = itot;
    int idist = itot;
    int jdist = 1;

    fftwf_r2r_kind kindf[] = {FFTW_R2HC};
    fftwf_r2r_kind kindb[] = {FFTW_HC2R};
    fftwf_plan iplanf = fftwf_plan_many_r2r(rank, ni, jtot, fftini, ni, istride, idist,
                                fftouti, ni, istride, idist, kindf, FFTW_EXHAUSTIVE);
    fftwf_plan iplanb = fftwf_plan_many_r2r(rank, ni, jtot, fftini, ni, istride, idist,
                                fftouti, ni, istride, idist, kindb, FFTW_EXHAUSTIVE);
    fftwf_plan jplanf = fftwf_plan_many_r2r(rank, nj, itot, fftinj, nj, jstride, jdist,
                                fftoutj, nj, jstride, jdist, kindf, FFTW_EXHAUSTIVE);
    fftwf_plan jplanb = fftwf_plan_many_r2r(rank, nj, itot, fftinj, nj, jstride, jdist,
                                fftoutj, nj, jstride, jdist, kindb, FFTW_EXHAUSTIVE);

    char filename[256];
    std::sprintf(filename, "%s.%07d", "fftwplan", 0);

    int n = fftwf_export_wisdom_to_filename(filename);
    if (n == 0)
        throw std::runtime_error("Error saving FFTW plan");

    fftwf_forget_wisdom();

    ss_iplanf << fftwf_sprint_plan(iplanf);
    ss_iplanb << fftwf_sprint_plan(iplanb);
    ss_jplanf << fftwf_sprint_plan(jplanf);
    ss_jplanb << fftwf_sprint_plan(jplanb);
}

void load(
        const int itot, const int jtot,
        std::stringstream& ss_iplanf,
        std::stringstream& ss_iplanb,
        std::stringstream& ss_jplanf,
        std::stringstream& ss_jplanb)
{
    char filename[256];
    std::sprintf(filename, "%s.%07d", "fftwplan", 0);

    int n = fftwf_import_wisdom_from_filename(filename);
    if (n == 0)
        throw std::runtime_error("Error loading FFTW Plan");

    int rank = 1;
    int ni[] = {itot};
    int nj[] = {jtot};
    int istride = 1;
    int jstride = itot;
    int idist = itot;
    int jdist = 1;
    fftwf_r2r_kind kindf[] = {FFTW_R2HC};
    fftwf_r2r_kind kindb[] = {FFTW_HC2R};
    fftwf_plan iplanf = fftwf_plan_many_r2r(rank, ni, jtot, fftini, ni, istride, idist,
            fftouti, ni, istride, idist, kindf, FFTW_WISDOM_ONLY);
    fftwf_plan iplanb = fftwf_plan_many_r2r(rank, ni, jtot, fftini, ni, istride, idist,
            fftouti, ni, istride, idist, kindb, FFTW_WISDOM_ONLY);
    fftwf_plan jplanf = fftwf_plan_many_r2r(rank, nj, itot, fftinj, nj, jstride, jdist,
            fftoutj, nj, jstride, jdist, kindf, FFTW_WISDOM_ONLY);
    fftwf_plan jplanb = fftwf_plan_many_r2r(rank, nj, itot, fftinj, nj, jstride, jdist,
            fftoutj, nj, jstride, jdist, kindb, FFTW_WISDOM_ONLY);

    fftwf_forget_wisdom();

    ss_iplanf << fftwf_sprint_plan(iplanf);
    ss_iplanb << fftwf_sprint_plan(iplanb);
    ss_jplanf << fftwf_sprint_plan(jplanf);
    ss_jplanb << fftwf_sprint_plan(jplanb);
}

int main()
{
    const int itot = 64;
    const int jtot = 48;

    fftini  = fftwf_alloc_real(itot*jtot);
    fftouti = fftwf_alloc_real(itot*jtot);
    fftinj  = fftwf_alloc_real(jtot*itot);
    fftoutj = fftwf_alloc_real(jtot*itot);

    int n = 0;
    while (true)
    {
        std::cout << "Test " << ++n << std::endl;

        std::stringstream ss_iplanf_save;
        std::stringstream ss_iplanb_save;
        std::stringstream ss_jplanf_save;
        std::stringstream ss_jplanb_save;

        std::stringstream ss_iplanf_load;
        std::stringstream ss_iplanb_load;
        std::stringstream ss_jplanf_load;
        std::stringstream ss_jplanb_load;

        save(itot, jtot, ss_iplanf_save, ss_iplanb_save, ss_jplanf_save, ss_jplanb_save);
        load(itot, jtot, ss_iplanf_load, ss_iplanb_load, ss_jplanf_load, ss_jplanb_load);

        if (ss_iplanf_save.str() != ss_iplanf_load.str())
            std::cout << "iplanf not identical" << std::endl;
        if (ss_iplanb_save.str() != ss_iplanb_load.str())
            std::cout << "iplanb not identical" << std::endl;
        if (ss_jplanf_save.str() != ss_jplanf_load.str())
            std::cout << "jplanf not identical" << std::endl;
        if (ss_jplanb_save.str() != ss_jplanb_load.str())
            std::cout << "jplanb not identical" << std::endl;
    }

    return 0;
}
