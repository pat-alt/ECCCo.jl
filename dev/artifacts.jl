using ghr_jll
using LazyArtifacts
using LibGit2
using Pkg.Artifacts
using Serialization

# Artifacts:
artifact_toml = LazyArtifacts.find_artifacts_toml(".")

function generate_artifacts(
    datafiles;
    artifact_name="artifacts-$VERSION",
    root=".",
    artifact_toml=joinpath(root, "Artifacts.toml"),
    deploy=true,
    tag=nothing,
)
    if isnothing(tag)
        tag = replace(lowercase(artifact_name), " " => "-")
    end

    if deploy && !haskey(ENV, "GITHUB_TOKEN")
        @warn "For automatic github deployment, need GITHUB_TOKEN. Not found in ENV, attemptimg global git config."
    end

    if deploy
        # Where we will put our tarballs
        tempdir = mktempdir()

        # Try to detect where we should upload these weights to (or just override
        # as shown in the commented-out line)
        origin_url = get_git_remote_url(root)
        deploy_repo = "$(basename(dirname(origin_url)))/$(basename(origin_url))"
    end

    # Name for hash/artifact:
    artifact_name = create_artifact_name_from_path(datafiles, artifact_name)

    # create_artifact() returns the content-hash of the artifact directory once we're finished creating it
    hash = create_artifact() do artifact_dir
        cp(datafiles, joinpath(artifact_dir, artifact_name))
    end

    # Spit tarballs to be hosted out to local temporary directory:
    if deploy
        tarball_hash = archive_artifact(hash, joinpath(tempdir, "$(artifact_name).tar.gz"))

        # Calculate tarball url
        tarball_url = "https://github.com/$(deploy_repo)/releases/download/$(tag)/$(artifact_name).tar.gz"

        # Bind this to an Artifacts.toml file
        @info("Binding $(artifact_name) in Artifacts.toml...")
        bind_artifact!(
            artifact_toml,
            artifact_name,
            hash;
            download_info=[(tarball_url, tarball_hash)],
            lazy=true,
            force=true,
        )
    end

    if deploy
        # Upload tarballs to a special github release
        @info("Uploading tarballs to $(deploy_repo) tag `$(tag)`")

        ghr() do ghr_exe
            println(
                readchomp(
                    `$ghr_exe -replace -u $(dirname(deploy_repo)) -r $(basename(deploy_repo)) $(tag) $(tempdir)`,
                ),
            )
        end

        @info("Artifacts.toml file now contains all bound artifact names")
    end
end

function get_git_remote_url(repo_path::String=".")
    repo = LibGit2.GitRepo(repo_path)
    origin = LibGit2.get(LibGit2.GitRemote, repo, "origin")
    return LibGit2.url(origin)
end
