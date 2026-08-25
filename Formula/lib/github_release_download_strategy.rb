require "download_strategy"
require "utils/github"

class DwellGitHubReleaseDownloadStrategy < CurlDownloadStrategy
  def initialize(url, name, version, **meta)
    super

    match = url.match(%r{\Ahttps://github\.com/([^/]+)/([^/]+)/releases/download/([^/]+)/([^/]+)\z})
    raise CurlDownloadStrategyError, "Invalid GitHub release asset URL." unless match

    @owner, @repo, @tag, @filename = match.captures
  end

  private

  def _fetch(url:, resolved_url:, timeout:)
    token = ENV["HOMEBREW_GITHUB_API_TOKEN"]
    return super if token.blank?

    release = GitHub.get_release(@owner, @repo, @tag)
    asset = release.fetch("assets").find { |candidate| candidate["name"] == @filename }
    raise CurlDownloadStrategyError, "GitHub release asset #{@filename} was not found." unless asset

    curl_download "https://api.github.com/repos/#{@owner}/#{@repo}/releases/assets/#{asset.fetch("id")}",
                  "--header", "Accept: application/octet-stream",
                  "--header", "Authorization: Bearer #{token}",
                  to:      temporary_path,
                  timeout: timeout
  end
end
