#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Platform {
    Unknown,
    None,
    GitHub,
    GitLab,
    AzureDevOps,
    Gitea,
    Bitbucket,
}

impl Platform {
    pub fn as_str(&self) -> &'static str {
        match self {
            Platform::Unknown => "unknown",
            Platform::None => "none",
            Platform::GitHub => "github",
            Platform::GitLab => "gitlab",
            Platform::AzureDevOps => "azuredevops",
            Platform::Gitea => "gitea",
            Platform::Bitbucket => "bitbucket",
        }
    }
}

pub fn platform_from_string(s: &str) -> Result<Platform, String> {
    match s.to_lowercase().as_str() {
        "" | "unknown" => Ok(Platform::Unknown),
        "none" => Ok(Platform::None),
        "github" => Ok(Platform::GitHub),
        "gitlab" => Ok(Platform::GitLab),
        "azuredevops" => Ok(Platform::AzureDevOps),
        "gitea" => Ok(Platform::Gitea),
        "bitbucket" => Ok(Platform::Bitbucket),
        _ => Err(format!("invalid scm platform value: {}", s)),
    }
}
